"""
Foundation Model Comparison — MobileNetV2 vs 1D CNN
==================================================
Fine-tunes a pre-trained MobileNetV2 (ImageNet weights) on the credit-card
fraud dataset and compares performance with the custom 1D CNN.

Usage:
    python run_foundation.py

Default parameters:
    --data_path    data/creditcard.csv
    --cnn_ckpt     models/best_model.keras
    --epochs       15
    --batch_size   1024
    --lr           0.001
    --seed         42
    --out_dir      outputs/
"""

import argparse
import os
import json
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
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras

from src.foundation_model import build_foundation_model, prepare_foundation_data
from src.model import build_cnn_model
from src.utils import evaluate_model
from src.dataloader import load_dataset


def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Data preparation -----
    df = load_dataset(args.data_path)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=args.seed, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # ----- 1D CNN (load checkpoint or train) -----
    X_tr_cnn = X_train_sm.reshape((X_train_sm.shape[0], X_train_sm.shape[1], 1))
    X_te_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    if args.cnn_ckpt and os.path.exists(args.cnn_ckpt):
        print(f"Loading CNN from {args.cnn_ckpt}...")
        cnn_model = keras.models.load_model(args.cnn_ckpt)
    else:
        print("Training 1D CNN...")
        cnn_model = build_cnn_model(input_shape=(X_tr_cnn.shape[1], 1))
        cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        cnn_model.fit(
            X_tr_cnn, y_train_sm,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                ),
            ],
            verbose=1,
        )

    print("\n--- 1D CNN Evaluation ---")
    cnn_metrics = evaluate_model(cnn_model, X_te_cnn, y_test, model_name="1D CNN")

    # ----- Foundation Model: MobileNetV2 -----
    print("\n--- Training Foundation Model (MobileNetV2) ---")
    # Use class_weight instead of SMOTE to avoid inflating dataset size
    # (keeps training fast while still handling class imbalance)
    X_tr_img, X_te_img = prepare_foundation_data(X_train, X_test)

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: cw[0], 1: cw[1]}
    print(f"  Using class_weight: {{0: {cw[0]:.2f}, 1: {cw[1]:.2f}}}")

    fm_model = build_foundation_model(input_shape=(6, 5, 3))
    fm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    fm_model.fit(
        X_tr_img, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
        ],
        verbose=1,
    )

    print("\n--- Foundation Model (MobileNetV2) Evaluation ---")
    fm_metrics = evaluate_model(fm_model, X_te_img, y_test,
                                model_name="MobileNetV2 (Foundation)")

    # ----- Comparison -----
    rows = []
    for name, m in [("1D CNN", cnn_metrics),
                     ("MobileNetV2 (Foundation)", fm_metrics)]:
        rows.append({
            'Model': name,
            'Precision': round(m['precision'], 4),
            'Recall': round(m['recall'], 4),
            'F1': round(m['f1'], 4),
            'ROC-AUC': round(m['roc_auc'], 4),
            'PR-AUC': round(m['pr_auc'], 4),
        })

    comp_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'foundation_comparison.csv')
    comp_df.to_csv(csv_path, index=False)

    print(f"\n{'='*65}")
    print("  1D CNN vs FOUNDATION MODEL (MobileNetV2)")
    print(f"{'='*65}")
    print(comp_df.to_string(index=False))
    print(f"\n  Saved: {csv_path}")

    # ----- Bar chart -----
    metrics_to_plot = ['F1', 'ROC-AUC', 'PR-AUC']
    x = np.arange(len(metrics_to_plot))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    cnn_vals = [comp_df.iloc[0][m] for m in metrics_to_plot]
    fm_vals = [comp_df.iloc[1][m] for m in metrics_to_plot]

    bars1 = ax.bar(x - width / 2, cnn_vals, width, label='1D CNN',
                   color='#ff7c00', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, fm_vals, width, label='MobileNetV2 (Foundation)',
                   color='#1ac938', edgecolor='white', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('1D CNN vs Foundation Model (MobileNetV2)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, 'foundation_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # ----- JSON -----
    json_out = []
    for name, m in [("1D CNN", cnn_metrics),
                     ("MobileNetV2 (Foundation)", fm_metrics)]:
        jr = {k: v for k, v in m.items() if k != 'confusion_matrix'}
        jr['model_name'] = name
        json_out.append(jr)
    json_path = os.path.join(args.out_dir, 'foundation_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"  JSON saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Foundation Model (MobileNetV2) vs CNN comparison")
    parser.add_argument("--data_path", type=str, default="data/creditcard.csv")
    parser.add_argument("--cnn_ckpt", type=str, default="models/best_model.keras")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/")
    args = parser.parse_args()
    main(args)
