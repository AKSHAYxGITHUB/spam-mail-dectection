import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ModelPipeline:
    def __init__(self, representation='tfidf'):
        self.representation = representation
        self.vectorizer = None
        self.model = None

    def build_features(self, X_train, X_test=None):
        if self.representation == 'tfidf':
            print("Using TF-IDF Representation...")
            self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
            
            if X_test is not None:
                X_test_vec = self.vectorizer.transform(X_test).toarray()
                return X_train_vec, X_test_vec
            return X_train_vec
            
        elif self.representation == 'sentence_transformers':
            print("Using Sentence Transformers...")
            try:
                from sentence_transformers import SentenceTransformer
                self.vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("sentence-transformers not installed. Falling back to TF-IDF.")
                self.representation = 'tfidf'
                return self.build_features(X_train, X_test)
                
            X_train_vec = self.vectorizer.encode(X_train)
            if X_test is not None:
                X_test_vec = self.vectorizer.encode(X_test)
                return X_train_vec, X_test_vec
            return X_train_vec
        else:
            raise ValueError("Unsupported representation. Choose 'tfidf' or 'sentence_transformers'.")

    def build_ensemble_model(self):
        print("Building Ensemble Model (Logistic Regression + Random Forest + XGBoost)...")
        # Initialize models
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=20, max_depth=25, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, n_estimators=20)

        # Create Voting Ensemble
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            self.build_ensemble_model()
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # roc_auc needs probabilities
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = 0.0
            
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
        
        print("\n--- Model Evaluation ---")
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print(f"{k.capitalize()}: {v:.4f}")
        print("Confusion Matrix:")
        print(np.array(metrics['confusion_matrix']))
        
        # Save metrics
        with open('evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics

    def save_artifacts(self, model_path='spam_model.pkl', vec_path='vectorizer.pkl'):
        print(f"Saving model to {model_path} and vectorizer to {vec_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        if self.representation == 'tfidf':
            with open(vec_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        print("Artifacts saved successfully.")

    def export_to_code(self, output_path='model_code.py'):
        """Converts the model to pure Python code using m2cgen."""
        print(f"Exporting model to pure Python code: {output_path}...")
        try:
            import m2cgen as m2c
            from sklearn.ensemble import VotingClassifier
            import sys
            
            # Increase recursion limit for deep trees
            sys.setrecursionlimit(5000)
            
            if isinstance(self.model, VotingClassifier):
                print("Detected VotingClassifier. Exporting individual estimators...")
                sub_model_files = []
                for name, est in self.model.named_estimators_.items():
                    sub_path = f"model_{name}.py"
                    print(f"Exporting {name} to {sub_path}...")
                    code = m2c.export_to_python(est)
                    with open(sub_path, 'w') as f:
                        f.write(f"# Generated by m2cgen - {name}\n")
                        f.write(code)
                    sub_model_files.append((name, sub_path))
                
                # Create a master wrapper
                with open(output_path, 'w') as f:
                    f.write("# Master Ensemble Wrapper (Zero Dependencies)\n")
                    for name, sub_path in sub_model_files:
                        f.write(f"from {sub_path[:-3]} import score as score_{name}\n")
                    
                    f.write("\ndef score(input):\n")
                    f.write("    # Simple soft voting implementation\n")
                    f.write("    scores = [\n")
                    for name, _ in sub_model_files:
                        f.write(f"        score_{name}(input),\n")
                    f.write("    ]\n")
                    f.write("    # Return average score\n")
                    f.write("    return sum(scores) / len(scores)\n")
                
                print("Ensemble exported successfully.")
                return True
            else:
                code = m2c.export_to_python(self.model)
                with open(output_path, 'w') as f:
                    f.write("# Generated by m2cgen (Zero Dependencies)\n")
                    f.write(code)
                print("Model code exported successfully.")
                return True
        except Exception as e:
            print(f"Error exporting model code: {e}")
            import traceback
            traceback.print_exc()
            return False

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
                "sublinear_tf": self.vectorizer.sublinear_tf
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print("Vectorizer JSON saved.")
        return True

if __name__ == '__main__':
    # Simple test
    print("ModelPipeline module loaded.")
