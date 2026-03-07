import m2cgen as m2c
from sklearn.linear_model import LogisticRegression
import numpy as np

print("Testing m2cgen...")
model = LogisticRegression()
X = np.array([[1, 2], [3, 4]])
y = np.array([0, 1])
model.fit(X, y)

print("Exporting...")
try:
    code = m2c.export_to_python(model)
    print("Success!")
    print(code[:100])
except Exception as e:
    print(f"Error: {e}")
