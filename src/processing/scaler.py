from sklearn.preprocessing import StandardScaler


def scale_data(X_train, X_test):
    print("\n[Scaler] Scaling data...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler