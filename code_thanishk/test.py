"""
Credit Card Fraud Detection - Test/Evaluation Script (Optional)

Loads a saved checkpoint and evaluates on the held-out test set.
This is optional since train.py already prints test-set metrics.
Useful for standalone verification of a saved model without retraining.

Usage:
    python test.py

Default parameters:
    --data_path    data/creditcard.csv
    --ckpt         models/best_model.keras
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from src.utils import evaluate_model
from src.dataloader import load_dataset


def main(args):
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data (auto-downloads via kagglehub if CSV not found)
    df = load_dataset(args.data_path)
    print(f"Dataset shape: {df.shape}")

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Same preprocessing as training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Same stratified split to reproduce the test set
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Reshape for CNN
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = keras.models.load_model(args.ckpt)

    # Evaluate
    print("\n--- Test Set Evaluation ---")
    evaluate_model(model, X_test_cnn, y_test, model_name="1D CNN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN for fraud detection")
    parser.add_argument("--data_path", type=str, default="data/creditcard.csv")
    parser.add_argument("--ckpt", type=str, default="models/best_model.keras")
    args = parser.parse_args()
    main(args)
