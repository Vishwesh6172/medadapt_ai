import pandas as pd
import matplotlib.pyplot as plt


def plot_results():
    df = pd.read_csv("outputs/results.csv")

    # Accuracy graph
    plt.figure()
    plt.plot(df["Accuracy"], marker='o')
    plt.title("Accuracy Across Runs")
    plt.xlabel("Run Index")
    plt.ylabel("Accuracy")
    plt.savefig("outputs/accuracy_trend.png")
    plt.close()

    # AUC graph
    plt.figure()
    plt.plot(df["AUC"], marker='o')
    plt.title("AUC Across Runs")
    plt.xlabel("Run Index")
    plt.ylabel("AUC")
    plt.savefig("outputs/auc_trend.png")
    plt.close()

    print("[Compare] Graphs saved in outputs/")