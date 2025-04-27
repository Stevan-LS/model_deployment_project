import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, make_scorer, log_loss
import pickle

def train_log_reg(X,y):
    # Configuration de la grille des hyperparamètres pour LogisticRegression
    log_reg = LogisticRegression(C=0.1,max_iter=1000, random_state=42)

    # Entraînement
    log_reg.fit(X, y)

    return log_reg

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

