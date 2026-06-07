from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    print("\n[SMOTE] Applying SMOTE...")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("[SMOTE] Before:", X.shape)
    print("[SMOTE] After :", X_res.shape)

    # Check balance
    import numpy as np
    unique, counts = np.unique(y_res, return_counts=True)
    print("[SMOTE] Class distribution:", dict(zip(unique, counts)))

    return X_res, y_res