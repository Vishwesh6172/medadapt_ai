import numpy as np
from src.deploy.load_model import load_best_model
from src.deploy.input_config import get_categorical_mappings
from src.model.ensemble import ensemble_predict_proba


def interpret_prediction(prob):
    if prob < 0.3:
        return "✅ Low risk of diabetes"
    elif prob < 0.5:
        return "⚠ Mild risk — lifestyle improvement suggested"
    elif prob < 0.7:
        return "⚠ Moderate risk — consult doctor"
    else:
        return "🚨 High risk — medical attention recommended"


def get_user_input(feature_names):
    mappings = get_categorical_mappings()

    print("\nEnter patient details:\n")

    values = []

    for fname in feature_names:

        # Dropdown for categorical
        if fname in mappings:
            options = mappings[fname]

            print(f"\n{fname} options:")
            for i, opt in enumerate(options):
                print(f"{i}: {opt}")

            while True:
                try:
                    choice = int(input(f"Select {fname}: "))
                    if 0 <= choice < len(options):
                        values.append(choice)
                        break
                except:
                    pass
                print("Invalid selection. Try again.")

        else:
            # Numeric input
            while True:
                try:
                    val = float(input(f"{fname}: "))
                    values.append(val)
                    break
                except:
                    print("Invalid input, try again.")

    return np.array(values).reshape(1, -1)


def main():
    bundle = load_best_model()

    model1 = bundle["model1"]
    model2 = bundle["model2"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    threshold = bundle["threshold"]

    print("\n[Deploy] Model loaded successfully")

    X = get_user_input(feature_names)

    # Apply scaler
    if scaler is not None:
        X = scaler.transform(X)

    # 🔥 FIX: use ensemble (same as training)
    prob = ensemble_predict_proba([model1, model2], [0.6, 0.4], X)[0]

    pred = int(prob >= threshold)

    print("\n==============================")
    print(f"Prediction: {'Diabetic' if pred else 'Non-Diabetic'}")
    print(f"Probability: {prob:.3f}")
    print(f"Threshold: {threshold:.3f}")
    print(interpret_prediction(prob))
    print("==============================")


if __name__ == "__main__":
    main()