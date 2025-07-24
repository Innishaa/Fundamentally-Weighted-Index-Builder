# data_utils.py
import numpy as np
import pandas as pd

def compute_fundamental_scores(latest_data):
    # Dummy scoring: Replace with real scoring like P/E, P/B, etc.
    return np.random.rand(len(latest_data))

def get_sector_matrix(security_names):
    # Dummy 3-sector one-hot encoding
    n = len(security_names)
    np.random.seed(42)
    sector_labels = np.random.choice([0, 1, 2], size=n)
    matrix = np.zeros((n, 3))
    matrix[np.arange(n), sector_labels] = 1
    return
