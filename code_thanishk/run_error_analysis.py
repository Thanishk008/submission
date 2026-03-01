"""
Error & Failure Analysis (Section 10)
=======================================
Analyzes false positives and false negatives from the 1D CNN model.
Produces:
  - Confusion matrix heatmap
  - Feature distribution comparison (correct vs misclassified)
  - Top misclassified sample details
  - Summary statistics of error patterns

Usage:
    python run_error_analysis.py

Default parameters:
    --data_path    data/creditcard.csv
    --ckpt         models/best_model.keras
    --seed         42
    --out_dir      outputs/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
sns.set_palette("bright")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras

from src.model import build_cnn_model
from src.dataloader import load_dataset


def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Data -----
    df = load_dataset(args.data_path)
    feature_names = [c for c in df.columns if c != 'Class']
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # ----- Load or train model -----
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading model from {args.ckpt}...")
        model = keras.models.load_model(args.ckpt)
    else:
        print("No checkpoint found — training CNN from scratch...")
        smote = SMOTE(random_state=args.seed, k_neighbors=5)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)
        X_tr_cnn = X_tr_sm.reshape((X_tr_sm.shape[0], X_tr_sm.shape[1], 1))

        model = build_cnn_model(input_shape=(X_tr_cnn.shape[1], 1))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy', metrics=['accuracy'],
        )
        model.fit(
            X_tr_cnn, y_tr_sm, batch_size=32, epochs=50,
            validation_split=0.2,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=1,
        )

    # ----- Predictions -----
    y_proba = model.predict(X_test_cnn, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred,
                                target_names=['Legitimate', 'Fraud']))

    # ----- Categorize predictions -----
    tp_mask = (y_test == 1) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)  # missed fraud
    fp_mask = (y_test == 0) & (y_pred == 1)  # false alarm
    tn_mask = (y_test == 0) & (y_pred == 0)

    print(f"\nTrue Positives  (correct fraud):     {tp_mask.sum():,}")
    print(f"False Negatives (missed fraud):       {fn_mask.sum():,}")
    print(f"False Positives (false alarm):        {fp_mask.sum():,}")
    print(f"True Negatives  (correct legit):      {tn_mask.sum():,}")

    # ========================================================
    # 1. Confusion Matrix Heatmap
    # ========================================================
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='coolwarm',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'], ax=ax)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title('Confusion Matrix — 1D CNN', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ========================================================
    # 2. Confidence distribution for errors
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # False Negatives — model was confident it's legit but it's fraud
    if fn_mask.sum() > 0:
        axes[0].hist(y_proba[fn_mask], bins=20, color='#e8000b',
                     edgecolor='white', alpha=0.8)
        axes[0].set_title(f'False Negatives (n={fn_mask.sum()})', fontweight='bold')
        axes[0].set_xlabel('Predicted P(Fraud)')
        axes[0].set_ylabel('Count')
        axes[0].axvline(0.5, color='white', linestyle='--', alpha=0.7)
    else:
        axes[0].text(0.5, 0.5, 'No False Negatives', ha='center',
                     va='center', transform=axes[0].transAxes)

    # False Positives — model flagged legit as fraud
    if fp_mask.sum() > 0:
        axes[1].hist(y_proba[fp_mask], bins=20, color='#023eff',
                     edgecolor='white', alpha=0.8)
        axes[1].set_title(f'False Positives (n={fp_mask.sum()})', fontweight='bold')
        axes[1].set_xlabel('Predicted P(Fraud)')
        axes[1].set_ylabel('Count')
        axes[1].axvline(0.5, color='white', linestyle='--', alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, 'No False Positives', ha='center',
                     va='center', transform=axes[1].transAxes)

    plt.suptitle('Prediction Confidence of Misclassified Samples', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.out_dir, 'error_confidence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ========================================================
    # 3. Feature distributions — correct vs misclassified fraud
    # ========================================================
    # Compare TP (correctly caught fraud) vs FN (missed fraud) on key features
    top_features = ['V14', 'V17', 'V12', 'V10', 'Amount', 'V4']
    available = [f for f in top_features if f in feature_names]

    if available and (tp_mask.sum() > 0 or fn_mask.sum() > 0):
        n_feat = len(available)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, feat in enumerate(available):
            idx = feature_names.index(feat)
            ax = axes[i]
            if tp_mask.sum() > 0:
                ax.hist(X_test[tp_mask, idx], bins=30, alpha=0.7,
                        label='TP (caught)', color='#1ac938', edgecolor='white')
            if fn_mask.sum() > 0:
                ax.hist(X_test[fn_mask, idx], bins=30, alpha=0.7,
                        label='FN (missed)', color='#e8000b', edgecolor='white')
            ax.set_title(feat, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Feature Distributions: Caught Fraud vs Missed Fraud',
                     fontweight='bold', fontsize=13)
        plt.tight_layout()
        path = os.path.join(args.out_dir, 'error_feature_dist.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ========================================================
    # 4. Summary statistics of misclassified samples
    # ========================================================
    summary_rows = []

    for mask, label in [(fn_mask, 'False Negative (missed fraud)'),
                        (fp_mask, 'False Positive (false alarm)')]:
        if mask.sum() == 0:
            continue
        subset = X_test[mask]
        summary_rows.append({
            'Error Type': label,
            'Count': int(mask.sum()),
            'Mean Confidence': round(float(y_proba[mask].mean()), 4),
            'Median Confidence': round(float(np.median(y_proba[mask])), 4),
            'Amount Mean': round(float(subset[:, feature_names.index('Amount')].mean()), 2)
                          if 'Amount' in feature_names else None,
            'Amount Median': round(float(np.median(subset[:, feature_names.index('Amount')])), 2)
                            if 'Amount' in feature_names else None,
        })

    if summary_rows:
        err_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(args.out_dir, 'error_analysis_summary.csv')
        err_df.to_csv(csv_path, index=False)
        print(f"\n{'='*65}")
        print("  ERROR ANALYSIS SUMMARY")
        print(f"{'='*65}")
        print(err_df.to_string(index=False))
        print(f"\n  Saved: {csv_path}")

    # ========================================================
    # 5. Sample-level error details (top 20 most confident errors)
    # ========================================================
    error_mask = fn_mask | fp_mask
    if error_mask.sum() > 0:
        err_indices = np.where(error_mask)[0]
        err_confidences = np.abs(y_proba[error_mask] - 0.5)  # distance from threshold
        top_k = min(20, len(err_indices))
        top_idx = err_indices[np.argsort(-err_confidences)[:top_k]]

        detail_rows = []
        for idx in top_idx:
            detail_rows.append({
                'Sample Index': idx,
                'True Label': 'Fraud' if y_test[idx] == 1 else 'Legit',
                'Predicted': 'Fraud' if y_pred[idx] == 1 else 'Legit',
                'P(Fraud)': round(float(y_proba[idx]), 4),
                'Error Type': 'FN' if y_test[idx] == 1 else 'FP',
            })

        detail_df = pd.DataFrame(detail_rows)
        csv_path = os.path.join(args.out_dir, 'error_top_samples.csv')
        detail_df.to_csv(csv_path, index=False)
        print(f"\n  Top {top_k} most confident errors:")
        print(detail_df.to_string(index=False))
        print(f"\n  Saved: {csv_path}")

    print("\n  Error analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error & failure analysis")
    parser.add_argument("--data_path", type=str, default="data/creditcard.csv")
    parser.add_argument("--ckpt", type=str, default="models/best_model.keras")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/")
    args = parser.parse_args()
    main(args)
