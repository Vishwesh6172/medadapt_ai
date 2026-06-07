from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_model(model, X_test, y_test, run_folder):
    print("\n[Evaluation] Evaluating model...")
    print(f"[Evaluation] Saving outputs to: {run_folder}")

    y_probs = model.predict_proba(X_test)[:, 1]

    threshold = 0.65
    y_pred = (y_probs > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", acc)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig(os.path.join(run_folder, "confusion_matrix.png"))
    plt.close()

    # ==============================
    # ROC CURVE
    # ==============================
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    print("\nAUC Score:", roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(os.path.join(run_folder, "roc_curve.png"))
    plt.close()

    return acc, roc_auc