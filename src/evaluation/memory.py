import json
import os


def update_memory(run_folder, accuracy):
    path = "outputs/memory.json"

    best = {"best_accuracy": 0, "best_run": ""}

    if os.path.exists(path):
        with open(path, "r") as f:
            best = json.load(f)

    if accuracy > best["best_accuracy"]:
        best["best_accuracy"] = accuracy
        best["best_run"] = run_folder

        with open(path, "w") as f:
            json.dump(best, f, indent=4)

        print(f"[Memory] New BEST model found: {accuracy}")
    else:
        print(f"[Memory] Best remains: {best['best_accuracy']}")