import numpy as np


def ensemble_predict_proba(models, weights, X):
    probs = np.zeros(len(X))

    for model, w in zip(models, weights):
        probs += w * model.predict_proba(X)[:, 1]

    return probs