import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_score = 0

    for t in np.linspace(0.3, 0.8, 50):
        preds = (y_probs > t).astype(int)
        score = f1_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"[Threshold] Best threshold: {round(best_thresh,3)} | F1: {round(best_score,3)}")

    return best_thresh