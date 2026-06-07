def encode_data(df):
    print("\n[Encoder] Encoding data...")

    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    return df