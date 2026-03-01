"""
Ablation Study & Baseline Comparison
======================================
Runs the Logistic Regression baseline, three ablation experiments,
and produces comparison tables + charts.

Baseline:    Logistic Regression vs 1D CNN
Technique 1: SMOTE  (train with vs without oversampling)
Technique 2: Dropout Rate  (0.0, 0.3, 0.5)
Technique 3: Batch Normalization  (with vs without)

Usage:
    python run_ablation.py

Default parameters:
    --data_path    data/creditcard.csv
    --epochs       30
    --batch_size   32
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

from src.model import build_cnn_model
from src.utils import evaluate_model
from src.baseline import train_baseline, evaluate_baseline
from src.dataloader import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_data(args):
    """Load + scale + stratified split. Returns raw arrays (no SMOTE yet)."""
    df = load_dataset(args.data_path)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_and_eval(X_train, y_train, X_test, y_test, *,
                   use_smote, dropout_rate, use_batchnorm,
                   epochs, batch_size, lr, seed, label):
    """Train one CNN variant and return metrics dict."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Optional SMOTE
    if use_smote:
        sm = SMOTE(random_state=seed, k_neighbors=5)
        X_tr, y_tr = sm.fit_resample(X_train, y_train)
    else:
        X_tr, y_tr = X_train.copy(), y_train.copy()

    # Reshape for 1D CNN
    X_tr_cnn = X_tr.reshape((X_tr.shape[0], X_tr.shape[1], 1))
    X_te_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_cnn_model(
        input_shape=(X_tr_cnn.shape[1], 1),
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"  SMOTE={use_smote}  Dropout={dropout_rate}  BatchNorm={use_batchnorm}")
    print(f"{'='*60}")

    model.fit(
        X_tr_cnn, y_tr,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = evaluate_model(model, X_te_cnn, y_test, model_name=label)
    metrics['label'] = label
    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def bar_chart(results, metric, title, save_path):
    """Simple grouped bar chart for one metric."""
    labels = [r['label'] for r in results]
    values = [r[metric] for r in results]
    colors = sns.color_palette("bright", len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(metric.upper())
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, min(max(values) + 0.1, 1.05))
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def summary_table(results, save_path):
    """Write a CSV + printed table of all metrics."""
    rows = []
    for r in results:
        rows.append({
            'Experiment': r['label'],
            'Precision': round(r['precision'], 4),
            'Recall': round(r['recall'], 4),
            'F1': round(r['f1'], 4),
            'ROC-AUC': round(r['roc_auc'], 4),
            'PR-AUC': round(r['pr_auc'], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\n  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data(args)

    all_results = []
    common = dict(epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, seed=args.seed)

    # ------------------------------------------------------------------
    # Baseline: Logistic Regression vs default CNN
    # ------------------------------------------------------------------
    print("\n>>> BASELINE COMPARISON")
    lr_model = train_baseline(X_train, y_train, seed=args.seed)
    lr_metrics = evaluate_baseline(lr_model, X_test, y_test,
                                   model_name="Logistic Regression")
    lr_metrics['label'] = 'Logistic Regression'
    all_results.append(lr_metrics)

    # Default CNN (with SMOTE, dropout=0.3, batchnorm=True)
    cnn_default = train_and_eval(
        X_train, y_train, X_test, y_test,
        use_smote=True, dropout_rate=0.3, use_batchnorm=True,
        label="1D CNN (default)", **common,
    )
    all_results.append(cnn_default)

    baseline_results = all_results[:2]
    bar_chart(baseline_results, 'f1', 'Baseline: LR vs CNN — F1',
              os.path.join(args.out_dir, 'baseline_comparison_f1.png'))
    bar_chart(baseline_results, 'roc_auc', 'Baseline: LR vs CNN — ROC-AUC',
              os.path.join(args.out_dir, 'baseline_comparison_roc.png'))

    # ------------------------------------------------------------------
    # Technique 1: SMOTE — with vs without
    # ------------------------------------------------------------------
    print("\n>>> TECHNIQUE 1: SMOTE")
    # Reuse cnn_default as "With SMOTE" (identical config) — only train Without SMOTE
    with_smote = {**cnn_default, 'label': 'With SMOTE'}
    without_smote = train_and_eval(
        X_train, y_train, X_test, y_test,
        use_smote=False, dropout_rate=0.3, use_batchnorm=True,
        label="Without SMOTE", **common,
    )
    all_results.extend([with_smote, without_smote])

    smote_results = [with_smote, without_smote]
    bar_chart(smote_results, 'f1', 'Technique 1: SMOTE — F1 Comparison',
              os.path.join(args.out_dir, 'ablation_smote_f1.png'))
    bar_chart(smote_results, 'roc_auc', 'Technique 1: SMOTE — ROC-AUC Comparison',
              os.path.join(args.out_dir, 'ablation_smote_roc.png'))

    # ------------------------------------------------------------------
    # Technique 2: Dropout Rate — 0.0, 0.3, 0.5
    # ------------------------------------------------------------------
    print("\n>>> TECHNIQUE 2: DROPOUT RATE")
    # Reuse cnn_default as Dropout=0.3 (identical config) — only train 0.0 and 0.5
    default_dropout = {**cnn_default, 'label': 'Dropout=0.3'}
    all_results.append(default_dropout)
    dropout_results = [default_dropout]
    for dr in [0.0, 0.5]:
        res = train_and_eval(
            X_train, y_train, X_test, y_test,
            use_smote=True, dropout_rate=dr, use_batchnorm=True,
            label=f"Dropout={dr}", **common,
        )
        all_results.append(res)
        dropout_results.append(res)
    # Sort by dropout value for consistent chart order
    dropout_results.sort(key=lambda r: float(r['label'].split('=')[1]))

    bar_chart(dropout_results, 'f1', 'Technique 2: Dropout Rate — F1 Comparison',
              os.path.join(args.out_dir, 'ablation_dropout_f1.png'))
    bar_chart(dropout_results, 'roc_auc', 'Technique 2: Dropout Rate — ROC-AUC Comparison',
              os.path.join(args.out_dir, 'ablation_dropout_roc.png'))

    # ------------------------------------------------------------------
    # Technique 3: Batch Normalization — with vs without
    # ------------------------------------------------------------------
    print("\n>>> TECHNIQUE 3: BATCH NORMALIZATION")
    # Reuse cnn_default as "With BatchNorm" (identical config) — only train Without BatchNorm
    with_bn = {**cnn_default, 'label': 'With BatchNorm'}
    without_bn = train_and_eval(
        X_train, y_train, X_test, y_test,
        use_smote=True, dropout_rate=0.3, use_batchnorm=False,
        label="Without BatchNorm", **common,
    )
    all_results.extend([with_bn, without_bn])
    bn_results = [with_bn, without_bn]

    bar_chart(bn_results, 'f1', 'Technique 3: BatchNorm — F1 Comparison',
              os.path.join(args.out_dir, 'ablation_batchnorm_f1.png'))
    bar_chart(bn_results, 'roc_auc', 'Technique 3: BatchNorm — ROC-AUC Comparison',
              os.path.join(args.out_dir, 'ablation_batchnorm_roc.png'))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_table(all_results, os.path.join(args.out_dir, 'ablation_summary.csv'))

    # Save raw JSON for later comparison scripts
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k != 'confusion_matrix'}
        json_results.append(jr)
    with open(os.path.join(args.out_dir, 'ablation_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  JSON saved: {os.path.join(args.out_dir, 'ablation_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for CNN fraud detection")
    parser.add_argument("--data_path", type=str, default="data/creditcard.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/")
    args = parser.parse_args()
    main(args)
