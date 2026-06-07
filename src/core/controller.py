def decide_pipeline(profile):
    print("\n[Controller] Generating strategy...")

    strategy = {}

    size = profile["size"]
    imbalance_ratio = profile["imbalance_ratio"]

    # ==============================
    # SIZE-BASED LOGIC
    # ==============================
    if size == "small":
        print("[Controller] SMALL dataset detected")
        strategy["use_gan"] = True
        strategy["shap_mode"] = "full"

    elif size == "medium":
        print("[Controller] MEDIUM dataset detected")
        strategy["use_gan"] = True
        strategy["shap_mode"] = "sample"

    else:
        print("[Controller] LARGE dataset detected")
        strategy["use_gan"] = False
        strategy["shap_mode"] = "sample"

    # ==============================
    # IMBALANCE LOGIC (INTELLIGENT)
    # ==============================
    if imbalance_ratio < 0.5:
        print("[Controller] STRONGLY IMBALANCED")
        strategy["use_smote"] = True

    elif imbalance_ratio < 0.8:
        print("[Controller] MILDLY IMBALANCED")
        strategy["use_smote"] = False   # 🔥 IMPORTANT CHANGE

    else:
        print("[Controller] BALANCED DATASET")
        strategy["use_smote"] = False

    print("\n[Controller] Final Strategy:", strategy)

    return strategy