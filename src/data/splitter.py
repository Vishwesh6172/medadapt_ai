from sklearn.model_selection import train_test_split


def split_data(df, target_column):
    print("\n[Splitter] Splitting data...")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 🔥 CRITICAL: unify to numpy
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test