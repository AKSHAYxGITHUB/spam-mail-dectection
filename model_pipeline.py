import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def _sigmoid(x):
    """Numerically stable sigmoid (works on scalars and numpy arrays)."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class ModelPipeline:
    """Trains a linear soft-voting ensemble (Logistic Regression + Linear SVM)
    on TF-IDF features and distills it to pure Python for zero-dependency
    serverless inference.
    """

    def __init__(self, representation='tfidf'):
        self.representation = representation
        self.vectorizer = None
        self.models = {}  # name -> fitted estimator

    def build_features(self, X_train, X_test=None):
        print("Using TF-IDF representation (word 1-2 grams, sublinear TF)...")
        # NOTE: keep this in sync with SimpleVectorizer in app.py — the runtime
        # re-implements this transform in pure Python, so the analyzer/token
        # pattern/sublinear_tf settings must match sklearn's defaults exactly.
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
        )
        # Keep matrices sparse — densifying 10k features over tens of thousands
        # of rows would blow up memory. LR and LinearSVC train on sparse input.
        X_train_vec = self.vectorizer.fit_transform(X_train)
        if X_test is not None:
            X_test_vec = self.vectorizer.transform(X_test)
            return X_train_vec, X_test_vec
        return X_train_vec

    def build_ensemble_model(self):
        print("Building linear soft-voting ensemble (LogisticRegression + LinearSVC)...")
        self.models = {
            'lr': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
            'svc': LinearSVC(C=1.0, random_state=42),
        }
        return self.models

    def train(self, X_train, y_train):
        if not self.models:
            self.build_ensemble_model()
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
        print("Training complete.")

    def ensemble_proba(self, X):
        """Spam probability = mean of each model's sigmoid(decision_function).

        This is exactly what the deployed pure-Python wrapper computes, so the
        evaluation metrics below reflect the model that actually ships.
        """
        probs = [_sigmoid(model.decision_function(X)) for model in self.models.values()]
        return np.mean(probs, axis=0)

    def evaluate(self, X_test, y_test):
        print("Evaluating ensemble (matches deployed inference)...")
        y_prob = self.ensemble_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_prob)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }

        print("\n--- Model Evaluation ---")
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print(f"{k.capitalize()}: {v:.4f}")
        print("Confusion Matrix:")
        print(np.array(metrics['confusion_matrix']))

        with open('evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics

    def save_artifacts(self, model_path='spam_model.pkl', vec_path='vectorizer.pkl'):
        print(f"Saving models to {model_path} and vectorizer to {vec_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(self.models, f)
        with open(vec_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Artifacts saved successfully.")

    def save_models_to_json(self, path='model_params.json'):
        """Exports each linear model's coefficients + intercept to JSON.

        Both estimators are linear, so inference is just a dot product
        (intercept + sum(coef * features)). Exporting the weights this way
        keeps the runtime dependency-free and — unlike m2cgen code generation —
        scales to any vocabulary size with no recursion limits or huge files.
        """
        print(f"Saving model parameters to {path}...")
        data = {}
        for name, model in self.models.items():
            # Binary classifiers expose coef_ of shape (1, n_features)
            data[name] = {
                "coef": model.coef_.ravel().tolist(),
                "intercept": float(model.intercept_[0]),
            }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(data)} linear models.")
        return True

    def save_vectorizer_to_json(self, path='vectorizer.json'):
        """Saves TF-IDF vectorizer parameters and vocabulary to JSON."""
        print(f"Saving vectorizer to {path}...")
        if not self.vectorizer:
            return False

        data = {
            "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
            "idf": self.vectorizer.idf_.tolist(),
            "params": {
                "ngram_range": list(self.vectorizer.ngram_range),
                "norm": self.vectorizer.norm,
                "use_idf": self.vectorizer.use_idf,
                "smooth_idf": self.vectorizer.smooth_idf,
                "sublinear_tf": self.vectorizer.sublinear_tf,
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print("Vectorizer JSON saved.")
        return True


if __name__ == '__main__':
    print("ModelPipeline module loaded.")
