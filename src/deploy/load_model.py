import os
import json
import joblib


def is_valid_bundle(bundle):
    return "model1" in bundle and "model2" in bundle


def load_best_model():
    memory_path = "outputs/memory.json"

    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            memory = json.load(f)

        best_run = memory.get("best_run", None)

        if best_run:
            model_path = os.path.join(best_run, "model_bundle.pkl")
            if os.path.exists(model_path):
                bundle = joblib.load(model_path)

                if is_valid_bundle(bundle):
                    print(f"[Deploy] Loading BEST model: {best_run}")
                    return bundle
                else:
                    print(f"[Deploy] Skipping old format model: {best_run}")

    # fallback search
    print("[Deploy] Searching for valid model...")

    runs = sorted(os.listdir("outputs"), reverse=True)

    for run in runs:
        if not run.startswith("run_"):
            continue

        path = os.path.join("outputs", run, "model_bundle.pkl")

        if os.path.exists(path):
            bundle = joblib.load(path)

            if is_valid_bundle(bundle):
                print(f"[Deploy] Using fallback model: outputs/{run}")
                return bundle

    raise FileNotFoundError("No valid (new-format) model found")