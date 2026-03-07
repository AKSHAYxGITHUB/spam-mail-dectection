import joblib
import m2cgen as m2c
import sys
import traceback

sys.setrecursionlimit(5000)

print("Loading model...")
model = joblib.load('spam_model.pkl')
print(f"Model type: {type(model)}")

try:
    from sklearn.ensemble import VotingClassifier
    if isinstance(model, VotingClassifier):
        print("Ensemble detected.")
        for name, est in model.named_estimators_.items():
            print(f"Exporting {name}...")
            code = m2c.export_to_python(est)
            print(f"Success for {name}! Length: {len(code)}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
