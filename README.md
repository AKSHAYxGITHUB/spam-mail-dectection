# Advanced AI Spam Detector 🛡️

A high-accuracy email filtering system powered by Machine Learning and NLP.

## 🚀 Live Demo
**[Try it out here!](https://spam-mail-dectection.vercel.app/)**

## 🌟 Key Features
- **Voting Classifier Ensemble**: Combines Logistic Regression and Random Forest for robust predictions (~91% accuracy).
- **Zero-Dependency Inference**: Runs on pure Python and manual TF-IDF implementation, bypassing heavy library overhead (no Scikit-learn or XGBoost required at runtime).
- **Infinite Scalability**: Optimized for Vercel Serverless Functions with an ultra-slim footprint (~45MB).
- **Advanced Explanation System**: Highlights suspicious words and provides confidence scores for transparent AI decisions.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **ML/DS**: Scikit-learn, XGBoost, NLTK, SMOTE (for training)
- **Frontend**: HTML5, Vanilla CSS3 (Modern dark-themed UI)
- **Deployment**: Vercel

## ⚙️ How it Works
1. **Preprocessing**: Cleans and lemmatizes input text using NLTK.
2. **Vectorization**: Transforms text into numerical features using a JSON-based TF-IDF transformer.
3. **Prediction**: Individual model scores are averaged to produce a final spam probability.
4. **Explanation**: The app identifies words with the highest TF-IDF scores to explain *why* a message was flagged.

## 📦 Zero-Dependency Optimization
To stay within Vercel's size limits, the trained models were converted to pure Python code using `m2cgen`, and the vectorizer parameters were exported to JSON. This allows the application to perform full AI inference using only standard Python libraries.

---
Developed by [Akshay P P](https://github.com/akshayxgithub)
