from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import re
from preprocessing import TextPreprocessor

app = Flask(__name__)

# Load models
try:
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model or vectorizer: {e}")
    model, vectorizer = None, None

preprocessor = TextPreprocessor()

def get_word_importance(text, vectorizer, max_words=5):
    """Finds the words with the highest TF-IDF score in the input text."""
    if not vectorizer:
        return []
    try:
        vec = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vec.toarray()[0]
        
        # Get indices of top scores
        top_indices = np.argsort(tfidf_scores)[::-1]
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

def highlight_words(text, words_to_highlight):
    """Highlights given words with HTML spans."""
    highlighted = text
    for w_obj in words_to_highlight:
        word = w_obj['word']
        # Use regex for whole word, replace with span
        highlighted = re.sub(
            r'\b(' + re.escape(word) + r')\b', 
            r'<span class="highlight-spam">\1</span>', 
            highlighted, 
            flags=re.IGNORECASE
        )
    return highlighted

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', result=None)

    email_content = request.form.get('email_content', '').strip()
    if not email_content:
        return render_template('index.html', result={"error": "Please enter an email."})

    if model is None or vectorizer is None:
        return render_template('index.html', result={"error": "Model not loaded. Please rain the model first."})

    # Preprocess text
    cleaned_text = preprocessor.clean_text(email_content)
    
    # Vectorize
    vec_text = vectorizer.transform([cleaned_text]).toarray()
    
    # Predict
    prediction = int(model.predict(vec_text)[0])
    probabilities = model.predict_proba(vec_text)[0]
    spam_prob = float(probabilities[1])
    ham_prob = float(probabilities[0])

    # Word importance
    top_words = get_word_importance(cleaned_text, vectorizer)
    
    # ---------------------------------------------------------
    # Rule-Based Heuristic Override
    # Ultra-short generic messages like "hi", "ok", "thanks" 
    # should default to HAM unless they contain blatant spam keywords
    # ---------------------------------------------------------
    word_count = len(re.findall(r'\b\w+\b', email_content))
    spam_keywords = ['win', 'prize', 'free', 'money', 'cash', 'urgent', 'click', 'link', 'buy', 'offer', 'viagra', 'lottery', 'guaranteed']
    contains_spam_keyword = any(keyword in email_content.lower() for keyword in spam_keywords)
    
    if word_count < 5 and not contains_spam_keyword:
        prediction = 0
        spam_prob = 0.05  # Artificial low probability for pure generic short text
        ham_prob = 0.95
        top_words = []
    
    # Determine result
    verdict = "Spam" if prediction == 1 else "Not Spam"
    confidence = spam_prob * 100 if prediction == 1 else ham_prob * 100
    
    # Only highlight if it's spam
    highlighted_text = highlight_words(email_content, top_words) if prediction == 1 else email_content
    
    # Generate Explanation
    if prediction == 1:
        words_list = ", ".join([w['word'] for w in top_words[:3]])
        explanation = f"This message exhibits strong spam characteristics, particularly the presence of suspicious keywords like: {words_list}."
    else:
        explanation = "This message appears to be normal and lacks typical spam indicators."

    result = {
        "original_text": email_content,
        "cleaned_text": cleaned_text,
        "verdict": verdict,
        "confidence": confidence,
        "spam_probability": spam_prob * 100,
        "top_words": top_words,
        "highlighted_text": highlighted_text,
        "explanation": explanation
    }

    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions."""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' parameter."}), 400
        
    cleaned_text = preprocessor.clean_text(data['text'])
    vec_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = int(model.predict(vec_text)[0])
    probabilities = model.predict_proba(vec_text)[0]
    
    return jsonify({
        "is_spam": bool(prediction == 1),
        "spam_probability": float(probabilities[1]),
        "ham_probability": float(probabilities[0])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)