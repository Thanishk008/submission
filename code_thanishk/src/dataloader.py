"""
Data loading utilities for Credit Card Fraud Detection
"""

import os
import glob
import pandas as pd
import kagglehub


def load_dataset(data_path=None):
    """Load the credit card fraud dataset.
    
    Tries the provided path first, then kagglehub download as fallback.
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Dataset loaded from {data_path}: {df.shape}")
        return df

    # Fallback: download via kagglehub
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in downloaded dataset")
    df = pd.read_csv(csv_files[0])
    print(f"Dataset downloaded via kagglehub: {df.shape}")
    return df
