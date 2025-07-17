import pandas as pd

def load_data(path='data/dummy_fundamentals.csv'):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df
