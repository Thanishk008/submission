"""
Utility functions for evaluation and visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Dark theme for all plots
plt.style.use('dark_background')
sns.set_palette("bright")

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a trained model and print metrics."""
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    pr_vals, re_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_val = auc(re_vals, pr_vals)

    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Results:")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}  PR-AUC: {pr_auc_val:.4f}")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'roc_auc': roc, 'pr_auc': pr_auc_val, 'confusion_matrix': cm
    }


def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss over Epochs', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy over Epochs', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    plt.close()
