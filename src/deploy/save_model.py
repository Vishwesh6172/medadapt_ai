import joblib
import os


def save_model(run_folder, model1, model2, scaler, feature_names, threshold):
    os.makedirs(run_folder, exist_ok=True)

    bundle = {
        "model1": model1,
        "model2": model2,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": threshold
    }

    path = os.path.join(run_folder, "model_bundle.pkl")
    joblib.dump(bundle, path)

    print(f"[Deploy] Model saved at: {path}")