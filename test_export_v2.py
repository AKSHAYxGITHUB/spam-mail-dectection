import joblib
import m2cgen as m2c
import sys

sys.setrecursionlimit(5000)

print("Loading model...")
model = joblib.load('spam_model.pkl')

try:
    from sklearn.ensemble import VotingClassifier
    if isinstance(model, VotingClassifier):
        lr_model = model.named_estimators_['lr']
        print("Exporting LR...")
        code = m2c.export_to_python(lr_model)
        with open("model_lr.py", "w") as f:
            f.write(code)
        print("Success for LR!")
except Exception as e:
    print(f"Error: {e}")
