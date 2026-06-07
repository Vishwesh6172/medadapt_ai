import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("\n[Loader] Data loaded:", df.shape)
    return df