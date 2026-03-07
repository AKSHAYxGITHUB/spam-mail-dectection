# Master Ensemble Wrapper (Zero Dependencies)
# Combining Logistic Regression and Random Forest
from model_lr import score as score_lr
from model_rf import score as score_rf

def score(input):
    # Simple soft voting implementation
    # lr_score is a float (probability)
    # rf_score is a list [prob_ham, prob_spam]
    
    s_lr = score_lr(input)
    s_rf = score_rf(input)[1] # index 1 is spam probability
    
    # Return average spam probability
    return (s_lr + s_rf) / 2
