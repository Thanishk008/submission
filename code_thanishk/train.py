"""
Credit Card Fraud Detection - Training Script
CNN-Based Deep Learning with SMOTE

Usage:
    python train.py

Default parameters:
    --data_path    data/creditcard.csv
    --epochs       50
    --batch_size   32
    --lr           0.001
    --seed         42
    --out_dir      outputs/
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model import build_cnn_model
from src.utils import evaluate_model, plot_training_history
from src.dataloader import load_dataset


def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load data (auto-downloads via kagglehub if CSV not found)
    df = load_dataset(args.data_path)
    print(f"Dataset shape: {df.shape}")

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split (70-30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # SMOTE
    smote = SMOTE(random_state=args.seed, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_smote.shape[0]:,} samples")

    # Reshape for 1D CNN
    X_train_cnn = X_train_smote.reshape((X_train_smote.shape[0], X_train_smote.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build model
    model = build_cnn_model(input_shape=(X_train_cnn.shape[1], 1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    # Train
    history = model.fit(
        X_train_cnn, y_train_smote,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "best_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Also save to models/ dir
    os.makedirs("models", exist_ok=True)
    model.save("models/best_model.keras")

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    evaluate_model(model, X_test_cnn, y_test, model_name="1D CNN")

    # Plot training history
    plot_training_history(history, save_path=os.path.join(args.out_dir, "training_history.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for fraud detection")
    parser.add_argument("--data_path", type=str, default="data/creditcard.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/")
    args = parser.parse_args()
    main(args)
