import pandas as pd


def remove_leakage_features(df, target_column):
    print("\n[Leakage Detector] Checking for leakage...")

    correlations = df.corr(numeric_only=True)[target_column].abs()

    # Drop target itself
    correlations = correlations.drop(target_column)

    # Threshold for suspicious correlation
    threshold = 0.95

    leakage_features = correlations[correlations > threshold].index.tolist()

    if leakage_features:
        print("[Leakage Detector] Removing leakage features:", leakage_features)
        df = df.drop(columns=leakage_features)
    else:
        print("[Leakage Detector] No leakage detected")

    return df