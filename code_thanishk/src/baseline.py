"""
Baseline Model for Credit Card Fraud Detection
Logistic Regression baseline for comparison with CNN.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve,
    confusion_matrix
)


def train_baseline(X_train, y_train, seed=42):
    """Train a Logistic Regression baseline model."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        class_weight='balanced',
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(model, X_test, y_test, model_name="Logistic Regression"):
    """Evaluate the baseline model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    pr_vals, re_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_val = auc(re_vals, pr_vals)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"  ROC-AUC:   {roc:.4f}  PR-AUC: {pr_auc_val:.4f}")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    return {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc,
        'pr_auc': pr_auc_val,
        'confusion_matrix': cm
    }
