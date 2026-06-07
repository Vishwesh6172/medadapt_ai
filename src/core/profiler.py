import pandas as pd


def profile_dataset(df, target_column):
    print("\n[Profiler] Analyzing dataset...")

    n_samples = df.shape[0]
    n_features = df.shape[1] - 1  # excluding target

    # Class distribution
    class_counts = df[target_column].value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()

    is_balanced = imbalance_ratio >= 0.8

    # Size category
    if n_samples < 2000:
        size = "small"
    elif n_samples < 50000:
        size = "medium"
    else:
        size = "large"

    profile = {
        "samples": n_samples,
        "features": n_features,
        "imbalance_ratio": imbalance_ratio,
        "balanced": is_balanced,
        "size": size
    }

    print("[Profiler] Profile:", profile)

    return profile