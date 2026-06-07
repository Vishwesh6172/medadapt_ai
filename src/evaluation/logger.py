import csv
import os
import time


def log_results(run_folder, dataset_type, accuracy, cv_score, auc_score, strategy):
    file_path = "outputs/results.csv"

    file_exists = os.path.isfile(file_path)

    for attempt in range(3):  # retry mechanism
        try:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow([
                        "Run",
                        "Dataset",
                        "Accuracy",
                        "CV Score",
                        "AUC",
                        "SMOTE",
                        "GAN",
                        "SHAP"
                    ])

                writer.writerow([
                    os.path.basename(run_folder),
                    dataset_type,
                    round(accuracy, 4),
                    round(cv_score, 4),
                    round(auc_score, 4),
                    strategy["use_smote"],
                    strategy["use_gan"],
                    strategy["shap_mode"]
                ])

            print("[Logger] Results saved")
            return

        except PermissionError:
            print("[Logger] File is open! Close results.csv and retrying...")
            time.sleep(2)

    print("[Logger] Failed to write results after retries.")