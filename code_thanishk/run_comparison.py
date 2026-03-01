"""
Unified Comparison — All Models
=================================
Loads results from ablation, baseline, and foundation model experiments
and produces a single comparison table + grouped bar chart.

Run the following scripts first to generate the JSON result files:
    python run_ablation.py
    python run_foundation.py

Usage:
    python run_comparison.py

Default parameters:
    --data_path    data/creditcard.csv
    --epochs       30
    --batch_size   32
    --lr           0.001
    --seed         42
    --out_dir      outputs/

If JSON files are not found, falls back to training all models from scratch.
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
from src.baseline import train_baseline, evaluate_baseline
from src.foundation_model import build_foundation_model, prepare_foundation_data
from src.utils import evaluate_model
from src.dataloader import load_dataset


def load_json_results(path):
    """Load a JSON results file if it exists."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def run_all_from_scratch(args):
    """Train and evaluate all models when no cached results exist."""
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = load_dataset(args.data_path)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    smote = SMOTE(random_state=args.seed, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    results = []

    # 1. Logistic Regression Baseline
    print("\n--- Logistic Regression Baseline ---")
    lr_model = train_baseline(X_train, y_train, seed=args.seed)
    lr_m = evaluate_baseline(lr_model, X_test, y_test, model_name="Logistic Regression")
    lr_m['model_name'] = 'Logistic Regression'
    results.append(lr_m)

    # 2. 1D CNN (default config)
    print("\n--- 1D CNN ---")
    X_tr_cnn = X_train_sm.reshape((X_train_sm.shape[0], X_train_sm.shape[1], 1))
    X_te_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    cnn_model = build_cnn_model(input_shape=(X_tr_cnn.shape[1], 1))
    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy', metrics=['accuracy'],
    )
    cnn_model.fit(
        X_tr_cnn, y_train_sm, batch_size=args.batch_size, epochs=args.epochs,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1,
    )
    cnn_m = evaluate_model(cnn_model, X_te_cnn, y_test, model_name="1D CNN")
    cnn_m['model_name'] = '1D CNN'
    results.append(cnn_m)

    # 3. Foundation Model (MobileNetV2)
    print("\n--- MobileNetV2 (Foundation) ---")
    X_tr_img, X_te_img = prepare_foundation_data(X_train_sm, X_test)
    fm_model = build_foundation_model(input_shape=(6, 5, 3))
    fm_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy', metrics=['accuracy'],
    )
    fm_model.fit(
        X_tr_img, y_train_sm, batch_size=args.batch_size, epochs=args.epochs,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=1,
    )
    fm_m = evaluate_model(fm_model, X_te_img, y_test,
                          model_name="MobileNetV2 (Foundation)")
    fm_m['model_name'] = 'MobileNetV2 (Foundation)'
    results.append(fm_m)

    return results


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Try loading cached results -----
    ablation_data = load_json_results(os.path.join(args.out_dir, 'ablation_results.json'))
    foundation_data = load_json_results(os.path.join(args.out_dir, 'foundation_results.json'))

    all_results = []

    if ablation_data and foundation_data:
        print("Loading cached results from JSON files...")

        # From ablation: pick Logistic Regression + 1D CNN (default)
        for r in ablation_data:
            label = r.get('label', r.get('model_name', ''))
            if label in ('Logistic Regression', '1D CNN (default)'):
                all_results.append({
                    'Model': label,
                    'Precision': r['precision'],
                    'Recall': r['recall'],
                    'F1': r['f1'],
                    'ROC-AUC': r['roc_auc'],
                    'PR-AUC': r['pr_auc'],
                })

        # Best ablation variant (highest F1 among non-baseline entries)
        ablation_variants = [r for r in ablation_data
                             if r.get('label', '') not in
                             ('Logistic Regression', '1D CNN (default)')]
        if ablation_variants:
            best = max(ablation_variants, key=lambda x: x['f1'])
            all_results.append({
                'Model': f"Best Ablation ({best.get('label', 'variant')})",
                'Precision': best['precision'],
                'Recall': best['recall'],
                'F1': best['f1'],
                'ROC-AUC': best['roc_auc'],
                'PR-AUC': best['pr_auc'],
            })

        # From foundation: pick MobileNetV2
        for r in foundation_data:
            name = r.get('model_name', '')
            if 'MobileNet' in name or 'Foundation' in name:
                all_results.append({
                    'Model': name,
                    'Precision': r['precision'],
                    'Recall': r['recall'],
                    'F1': r['f1'],
                    'ROC-AUC': r['roc_auc'],
                    'PR-AUC': r['pr_auc'],
                })
    else:
        print("No cached results — training all models from scratch...")
        raw = run_all_from_scratch(args)
        for r in raw:
            all_results.append({
                'Model': r['model_name'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1': r['f1'],
                'ROC-AUC': r['roc_auc'],
                'PR-AUC': r['pr_auc'],
            })

    # Round values
    for r in all_results:
        for k in ('Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'):
            r[k] = round(r[k], 4)

    # ----- Table -----
    comp_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.out_dir, 'unified_comparison.csv')
    comp_df.to_csv(csv_path, index=False)

    print(f"\n{'='*75}")
    print("  UNIFIED MODEL COMPARISON")
    print(f"{'='*75}")
    print(comp_df.to_string(index=False))
    print(f"\n  Saved: {csv_path}")

    # ----- Grouped Bar Chart -----
    metrics = ['F1', 'ROC-AUC', 'PR-AUC']
    models = comp_df['Model'].tolist()
    n_models = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = sns.color_palette("bright", n_colors=n_models)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        vals = [comp_df.loc[comp_df['Model'] == model, m].values[0] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors[i % len(colors)],
                      edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Unified Model Comparison', fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, 'unified_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # ----- Precision-Recall Table -----
    pr_metrics = ['Precision', 'Recall']
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x2 = np.arange(len(pr_metrics))
    for i, model in enumerate(models):
        vals = [comp_df.loc[comp_df['Model'] == model, m].values[0] for m in pr_metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax2.bar(x2 + offset, vals, width, label=model,
                       color=colors[i % len(colors)],
                       edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(pr_metrics, fontsize=11)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Precision & Recall Comparison', fontweight='bold', fontsize=13)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plot_path2 = os.path.join(args.out_dir, 'unified_precision_recall.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path2}")

    print("\n  Unified comparison complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified comparison of all models")
    parser.add_argument("--data_path", type=str, default="data/creditcard.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/")
    args = parser.parse_args()
    main(args)
