import numpy as np


def generate_synthetic_data(X, y, strategy):
    print("\n[GAN] Generating synthetic data...")

    # 🔥 FIX: ensure y is numpy (position-safe)
    if hasattr(y, "values"):
        y = y.values

    # ==============================
    # INTELLIGENT CONTROL
    # ==============================
    if strategy["size"] == "small":
        multiplier = 1.0
    elif strategy["size"] == "medium":
        multiplier = 0.5
    else:
        multiplier = 0.2

    n_samples = int(X.shape[0] * multiplier)

    # ==============================
    # GENERATE DATA
    # ==============================
    noise = np.random.normal(0, 0.02, (n_samples, X.shape[1]))

    idx = np.random.choice(X.shape[0], n_samples)

    synthetic = X[idx] + noise
    synthetic_labels = y[idx]

    X_new = np.vstack([X, synthetic])
    y_new = np.hstack([y, synthetic_labels])

    print(f"[GAN] Added {n_samples} samples")
    print(f"[GAN] Final dataset: {X_new.shape}")

    return X_new, y_new