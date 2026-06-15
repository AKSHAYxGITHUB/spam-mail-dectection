# Pure-Python linear ensemble inference (zero dependencies).
#
# Both models (Logistic Regression + Linear SVM) are linear, so scoring a
# sample is just intercept + sum(coef * features). We load their weights from
# model_params.json and average each model's sigmoid-squashed decision to get a
# final spam probability — exactly matching the training-time evaluation.
import json
import math
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DIR, 'model_params.json'), 'r') as _f:
    _MODELS = json.load(_f)  # {name: {"coef": [...], "intercept": float}}


def _sigmoid(x):
    # Numerically stable: avoid overflow for large-magnitude decisions
    if x < 0:
        z = math.exp(x)
        return z / (1.0 + z)
    return 1.0 / (1.0 + math.exp(-x))


def _decision(params, features):
    coef = params["coef"]
    total = params["intercept"]
    for i in range(len(coef)):
        total += coef[i] * features[i]
    return total


def score(features):
    """Returns the ensemble spam probability for a TF-IDF feature vector."""
    probs = [_sigmoid(_decision(p, features)) for p in _MODELS.values()]
    return sum(probs) / len(probs)
