# Advanced AI Spam Detector 🛡️

A spam filtering system for email/SMS messages, powered by Machine Learning and NLP.

## 🚀 Live Demo
**[Try it out here!](https://spam-mail-dectection.vercel.app/)**

## 📊 Performance
Evaluated on a held-out 20% test split (~8k messages) from a balanced corpus of ~195k emails:

| Metric    | Score |
|-----------|-------|
| Accuracy  | 96.7% |
| Precision | 95.8% |
| Recall    | 97.4% |
| F1        | 0.966 |
| ROC-AUC   | 0.996 |

## 🌟 Key Features
- **Hybrid ML + Signals**: A TF-IDF linear ensemble (Logistic Regression + Linear SVM) is combined with **SpamAssassin-style weighted heuristic signals** — URL/link analysis, link shorteners, suspicious TLDs, ALL-CAPS & punctuation, urgency/scam phrasing, money lures, and **sender-spoofing detection** from pasted headers. Signals are blended with the model in log-odds space.
- **Zero-Dependency Inference**: Runs on pure Python — model weights are exported as plain coefficient vectors and scored with a hand-written TF-IDF transform and dot product. No scikit-learn, NumPy, or other ML libraries at runtime.
- **Lightweight & Serverless**: The entire model (vocabulary + weights) is < 1 MB, ideal for Vercel Serverless Functions.
- **Explainable Predictions**: Shows the exact signals that fired (with their risk weight), the most influential words, the ML-only vs final score, and a plain-English reason.
- **Analytics Dashboard**: Spam-vs-legitimate breakdown, most-frequent-signal charts, and a history of analyzed messages (stored client-side).
- **JSON API**: `POST /api/predict` for programmatic access.

## 🛠️ Tech Stack
- **Runtime**: Python, Flask (pure-Python inference — no ML libraries)
- **Training**: scikit-learn, NLTK
- **Frontend**: HTML5, Vanilla CSS3 (modern dark-themed UI)
- **Deployment**: Vercel

## ⚙️ How it Works
1. **Preprocessing**: Cleans, normalizes, and lemmatizes input text using NLTK.
2. **Vectorization**: Transforms text into TF-IDF features (word 1–2 grams, sublinear TF) from a JSON-exported vocabulary / IDF table.
3. **ML prediction**: Each linear model's decision score is squashed with a sigmoid and the two are averaged (soft voting) to produce a base spam probability.
4. **Signal blending**: Engineered heuristic signals ([signals.py](signals.py)) each contribute a weight in *log-odds*; these are added to the model's log-odds and squashed back to a final probability. This mirrors how real filters (e.g. SpamAssassin) combine a Bayesian model with weighted rules.
5. **Explanation**: The app surfaces the fired signals, the most influential words, and the ML-vs-final score. Ultra-short generic messages with no signals default to "legitimate", since TF-IDF is unreliable on very few tokens.

## 📦 Zero-Dependency Optimization
Both models are linear, so a prediction is just `intercept + Σ(coef · feature)`. Instead of shipping scikit-learn to Vercel, the trained coefficients are exported to `model_params.json` and the TF-IDF vocabulary / IDF weights to `vectorizer.json`. The runtime ([model_code.py](model_code.py) + the `SimpleVectorizer` in [app.py](app.py)) re-implements TF-IDF and the dot product in ~60 lines of standard-library Python. The pure-Python path reproduces scikit-learn's output to machine precision, so the deployed model matches the evaluated one exactly.

## 🔌 API Usage
```bash
curl -X POST https://spam-mail-dectection.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! Claim your free prize now!"}'
```
Response:
```json
{
  "verdict": "Spam",
  "is_spam": true,
  "spam_probability": 94.0,
  "ml_probability": 88.2,
  "confidence": 94.0,
  "signals": [
    { "label": "Money / prize lure", "detail": "prize, claim your", "weight": 0.8 }
  ],
  "top_words": [{ "word": "prize", "score": 0.74 }],
  "explanation": "Flagged as spam due to money / prize lure, plus suspicious wording (prize)."
}
```
(Probabilities are percentages, 0–100.)

## 🧑‍💻 Local Development
```bash
# Runtime only (Flask + nltk)
pip install -r requirements.txt
python app.py                       # serves on http://localhost:5001

# To retrain the model from scratch (expects spam.csv with v1=label, v2=message)
pip install -r requirements-dev.txt
python train_model.py
```

---
Developed by [Akshay P P](https://github.com/akshayxgithub)
