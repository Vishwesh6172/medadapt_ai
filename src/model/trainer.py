from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def train_model(X, y, strategy):
    print("\n[Trainer] Selecting model...")

    # ==============================
    # MODEL SELECTION LOGIC
    # ==============================
    if strategy["size"] == "small":
        print("[Trainer] Using XGBoost (better for small data)")
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss'
        )

    else:
        print("[Trainer] Using RandomForest (stable for large data)")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

    print("\n[Trainer] Performing Cross Validation...")

    scores = cross_val_score(model, X, y, cv=5)

    print("[CV Scores]:", scores)
    print("[Mean CV]:", scores.mean())

    return model, scores.mean()