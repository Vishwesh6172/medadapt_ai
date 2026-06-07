import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os


def select_features_shap(X, y, feature_names, run_folder, mode="full", top_k=6):
    print("\n[SHAP] Selecting features...")

    # 🔥 FIX: ensure y is numpy array (position-safe)
    if hasattr(y, "values"):
        y = y.values

    # ==============================
    # SAMPLE MODE (for large data)
    # ==============================
    if mode == "sample":
        print("[SHAP] Using SAMPLE mode...")
        sample_size = min(2000, X.shape[0])
        idx = np.random.choice(X.shape[0], sample_size, replace=False)

        X_used = X[idx]
        y_used = y[idx]

    else:
        print("[SHAP] Using FULL mode...")
        X_used = X
        y_used = y

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_used, y_used)

    importance = model.feature_importances_

    indices = np.argsort(importance)[::-1][:top_k]

    print("[SHAP] Selected feature indices:", indices)

    selected_names = [feature_names[i] for i in indices]
    print("[SHAP] Selected features:", selected_names)

    # ==============================
    # PLOT
    # ==============================
    plt.figure()

    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [f"{feature_names[i]} ({i})" for i in sorted_idx]

    plt.bar(range(len(importance)), importance[sorted_idx])
    plt.xticks(range(len(importance)), sorted_names, rotation=45, ha='right')

    plt.title("Feature Importance")
    plt.tight_layout()

    plt.savefig(os.path.join(run_folder, "shap_importance.png"))
    plt.close()

    X_selected = X[:, indices]

    print("[SHAP] New shape:", X_selected.shape)

    return X_selected, indices