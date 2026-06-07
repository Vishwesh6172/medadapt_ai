import pandas as pd


def detect_target_column(df):
    print("\n[Target Detector] Detecting target column...")

    # Common names
    possible_targets = ["Outcome", "target", "label", "diabetes", "class"]

    for col in df.columns:
        if col.lower() in [t.lower() for t in possible_targets]:
            print(f"[Target Detector] Found target column: {col}")
            return col

    # Fallback: last column
    print("[Target Detector] No standard name found. Using last column.")
    return df.columns[-1]