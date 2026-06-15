from flask import Flask, render_template, request, jsonify
import json
import math
import re
from preprocessing import TextPreprocessor
import model_code
import signals

app = Flask(__name__)

class SimpleVectorizer:
    """Manual TF-IDF Vectorizer to avoid scikit-learn dependency."""
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.vocabulary = data['vocabulary']
        self.idf = data['idf']
        params = data['params']
        self.ngram_range = params['ngram_range']
        self.sublinear_tf = params.get('sublinear_tf', False)
        self.feature_names = sorted(self.vocabulary.keys(), key=lambda x: self.vocabulary[x])

    def transform(self, text):
        # Tokenize — match scikit-learn's default token pattern (2+ word chars)
        tokens = re.findall(r'\b\w\w+\b', text.lower())

        # Add n-grams
        if self.ngram_range[1] > 1:
            tokens_full = list(tokens)
            for n in range(2, self.ngram_range[1] + 1):
                for i in range(len(tokens) - n + 1):
                    tokens_full.append(" ".join(tokens[i:i+n]))
            tokens = tokens_full

        # Count frequencies (TF)
        tf = [0.0] * len(self.vocabulary)
        for t in tokens:
            if t in self.vocabulary:
                tf[self.vocabulary[t]] += 1

        # Sublinear TF scaling: 1 + log(tf) for non-zero counts (matches sklearn)
        if self.sublinear_tf:
            tf = [1.0 + math.log(c) if c > 0 else 0.0 for c in tf]

        # Multiply by IDF
        tfidf = [tf[i] * self.idf[i] for i in range(len(tf))]
        
        # Normalize (L2)
        norm = math.sqrt(sum(x*x for x in tfidf))
        if norm > 0:
            tfidf = [x / norm for x in tfidf]
            
        return tfidf

    def get_feature_names_out(self):
        return self.feature_names

# Load models
try:
    vectorizer = SimpleVectorizer('vectorizer.json')
    print("Zero-Dependency Vectorizer loaded. Model using code-based inference.")
except Exception as e:
    print(f"Warning: Could not load vectorizer: {e}")
    vectorizer = None

preprocessor = TextPreprocessor()

def get_word_importance(text, vectorizer, max_words=5):
    """Finds the words with the highest TF-IDF score in the input text."""
    if not vectorizer:
        return []
    try:
        tfidf_scores = vectorizer.transform(text)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get indices of top scores
        top_indices = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)
        top_words = []
        for idx in top_indices:
            if tfidf_scores[idx] > 0 and len(top_words) < max_words:
                top_words.append({
                    "word": feature_names[idx], 
                    "score": float(tfidf_scores[idx])
                })
        return top_words
    except Exception as e:
        print(f"Error in word importance: {e}")
        return []

def predict_spam(email_content):
    """Runs the full zero-dependency inference pipeline on raw email text.

    Combines the TF-IDF ML model with SpamAssassin-style engineered signals:
    the signal weights (in log-odds) are added to the model's log-odds before
    squashing back to a probability. Returns the prediction, probabilities,
    top contributing words, the fired signals, and the cleaned text. Shared by
    the web route and the JSON API so they can never diverge.
    """
    cleaned_text = preprocessor.clean_text(email_content)

    # 1) ML model probability (pure-Python TF-IDF + linear ensemble)
    vec_text = vectorizer.transform(cleaned_text)
    ml_prob = float(model_code.score(vec_text))
    top_words = get_word_importance(cleaned_text, vectorizer)

    # 2) Engineered signals -> log-odds adjustment
    fired_signals = signals.extract_signals(email_content)
    delta = signals.total_logodds(fired_signals)

    # 3) Blend in log-odds space, then squash back to a probability
    p = min(max(ml_prob, 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p)) + delta
    spam_prob = 1.0 / (1.0 + math.exp(-logit))

    # Short-message guard: too few tokens for reliable TF-IDF, and no signal
    # fired -> default to ham (e.g. "hi", "good morning").
    word_count = len(re.findall(r'\b\w+\b', email_content))
    if word_count < 5 and delta <= 0:
        spam_prob = 0.05
        top_words = []
        fired_signals = []

    ham_prob = 1.0 - spam_prob
    prediction = 1 if spam_prob >= 0.5 else 0

    return {
        "prediction": prediction,
        "spam_prob": spam_prob,
        "ham_prob": ham_prob,
        "ml_prob": ml_prob,
        "signal_logodds": delta,
        "top_words": top_words,
        "signals": fired_signals,
        "cleaned_text": cleaned_text,
    }


def build_result(email_content):
    """Builds the full result payload (verdict, confidence, signals,
    explanation) used by both the API and the front-end."""
    pred = predict_spam(email_content)
    is_spam = pred["prediction"] == 1
    spam_prob = pred["spam_prob"]
    confidence = spam_prob * 100 if is_spam else pred["ham_prob"] * 100

    # Human-readable explanation drawing on whichever evidence is strongest
    if is_spam:
        reasons = [s["label"].lower() for s in pred["signals"][:3]]
        words = [w["word"] for w in pred["top_words"][:3]]
        if reasons:
            explanation = "Flagged as spam due to " + ", ".join(reasons)
            if words:
                explanation += f", plus suspicious wording ({', '.join(words)})."
            else:
                explanation += "."
        elif words:
            explanation = ("Flagged as spam based on the language used, "
                           f"especially: {', '.join(words)}.")
        else:
            explanation = "Flagged as spam by the language model."
    else:
        explanation = "This message looks legitimate and lacks typical spam indicators."

    return {
        "original_text": email_content,
        "verdict": "Spam" if is_spam else "Not Spam",
        "is_spam": is_spam,
        "confidence": confidence,
        "spam_probability": spam_prob * 100,
        "ham_probability": pred["ham_prob"] * 100,
        "ml_probability": pred["ml_prob"] * 100,
        "top_words": pred["top_words"],
        "signals": pred["signals"],
        "explanation": explanation,
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions (used by the front-end and externally)."""
    if vectorizer is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' parameter."}), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({"error": "'text' parameter is empty."}), 400

    return jsonify(build_result(text))

if __name__ == '__main__':
    app.run(debug=True, port=5001)