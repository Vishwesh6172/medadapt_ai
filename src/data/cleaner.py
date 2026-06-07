def clean_data(df):
    print("\n[Cleaner] Cleaning data...")

    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    return df