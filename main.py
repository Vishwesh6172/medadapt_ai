from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.data.encoder import encode_data
from src.data.splitter import split_data

from src.core.profiler import profile_dataset
from src.core.controller import decide_pipeline
from src.core.target_detector import detect_target_column
from src.core.leakage_detector import remove_leakage_features

from src.processing.scaler import scale_data
from src.processing.smote import apply_smote
from src.processing.gan import generate_synthetic_data
from src.processing.shap import select_features_shap

from src.model.trainer import train_model
from src.model.ensemble import ensemble_predict_proba

from src.evaluation.threshold import find_best_threshold
from src.evaluation.report import generate_report
from src.evaluation.memory import update_memory
from src.evaluation.architecture import generate_architecture_diagram
from src.evaluation.logger import log_results

from src.deploy.save_model import save_model

import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# ==============================
# RUN FOLDER
# ==============================
def get_run_folder():
    os.makedirs("outputs", exist_ok=True)
    runs = [d for d in os.listdir("outputs") if d.startswith("run_")]
    nums = [int(r.split("_")[1]) for r in runs if r.split("_")[1].isdigit()]
    n = max(nums) + 1 if nums else 1
    path = f"outputs/run_{n:03d}"
    os.makedirs(path, exist_ok=True)
    return path


run_folder = get_run_folder()
print(f"[Main] Outputs will be saved in: {run_folder}")


# ==============================
# DATA
# ==============================
choice = input("Choose dataset (small / large): ").strip().lower()
df = load_data("data/large.csv") if choice == "large" else load_data("data/small.csv")

df = clean_data(df)
df = encode_data(df)

TARGET_COLUMN = detect_target_column(df)
df = remove_leakage_features(df, TARGET_COLUMN)

feature_names = df.drop(columns=[TARGET_COLUMN]).columns.tolist()


# ==============================
# PROFILE + STRATEGY
# ==============================
profile = profile_dataset(df, TARGET_COLUMN)
strategy = decide_pipeline(profile)
strategy["size"] = profile["size"]


# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN)


# ==============================
# SMOTE (before feature selection)
# ==============================
if strategy["use_smote"]:
    X_train, y_train = apply_smote(X_train, y_train)


# ==============================
# SHAP (FEATURE SELECTION FIRST)
# ==============================
X_train, selected_idx = select_features_shap(
    X_train,
    y_train,
    feature_names,
    run_folder,
    mode=strategy["shap_mode"]
)

X_test = X_test[:, selected_idx]


# ==============================
# SCALE (AFTER SHAP — FIXED)
# ==============================
X_train, X_test, scaler = scale_data(X_train, X_test)


# ==============================
# TRAIN MODELS
# ==============================
model1, cv_score = train_model(X_train, y_train, strategy)

model2 = RandomForestClassifier(n_estimators=150, random_state=42)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)


# ==============================
# GAN (OPTIONAL)
# ==============================
if strategy["use_gan"]:
    X_train, y_train = generate_synthetic_data(X_train, y_train, strategy)


# ==============================
# ENSEMBLE PREDICTION
# ==============================
probs = ensemble_predict_proba([model1, model2], [0.6, 0.4], X_test)


# ==============================
# THRESHOLD TUNING
# ==============================
best_thresh = find_best_threshold(y_test, probs)
preds = (probs > best_thresh).astype(int)


# ==============================
# EVALUATION
# ==============================
print("\n[Evaluation] Evaluating model...")

accuracy = accuracy_score(y_test, preds)
auc_score = roc_auc_score(y_test, probs)

print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, preds)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, preds))


# Save confusion matrix image
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig(os.path.join(run_folder, "confusion_matrix.png"))
plt.close()


# ==============================
# SAVE MODEL (CORRECTED)
# ==============================
selected_feature_names = [feature_names[i] for i in selected_idx]

save_model(
    run_folder,
    model1,
    model2,
    scaler,
    selected_feature_names,
    best_thresh
)


# ==============================
# REPORT + MEMORY + LOGGING
# ==============================
generate_report(run_folder, accuracy, auc_score)
update_memory(run_folder, accuracy)
log_results(run_folder, strategy["size"], accuracy, cv_score, auc_score, strategy)
generate_architecture_diagram(run_folder)


print("\n==============================")
print("FINAL TEST ACCURACY:", accuracy)
print("MEAN CV ACCURACY  :", cv_score)
print("AUC SCORE         :", auc_score)
print("OUTPUT FOLDER     :", run_folder)
print("==============================")