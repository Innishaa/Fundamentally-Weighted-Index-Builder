import pandas as pd
def load_data_from_excel(path):
    df = pd.read_excel(path, engine='openpyxl')
    return df

def load_data_from_csv(path):
    df = pd.read_csv(path)
    return df
    return df

def load_data(path='data/dummy_fundamentals.csv'):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df
