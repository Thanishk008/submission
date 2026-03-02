

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve, auc
)

from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier


SEED = 123
random.seed(SEED)
np.random.seed(SEED)

os.makedirs("results", exist_ok=True)


data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1).values
y = data["Class"].values

print("Dataset shape:", data.shape)
print("Fraud ratio:", sum(y)/len(y))


X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)


print("Before SMOTE:", np.bincount(y_train))

smote = SMOTE(random_state=SEED)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("After SMOTE:", np.bincount(y_train_smote))


tabnet = TabNetClassifier(
    seed=SEED,
    verbose=1
)

tabnet.fit(
    X_train_smote, y_train_smote,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric=['auc'],
    max_epochs=30,
    patience=5,
    batch_size=256
)


y_pred = tabnet.predict(X_test_scaled)
y_prob = tabnet.predict_proba(X_test_scaled)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# -------- PR AUC --------
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)

print("\n===== TabNet + SMOTE Results =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)
print("PR-AUC   :", pr_auc)


metrics_file = "results/model_metrics.csv"

metrics_dict = {
    "model": "TABNET_SMOTE",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc
}

metrics_df = pd.DataFrame([metrics_dict])

if os.path.exists(metrics_file):
    metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(metrics_file, index=False)

print("Metrics saved to results/model_metrics.csv")


cm = confusion_matrix(y_test, y_pred)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("TabNet Confusion Matrix")
plt.savefig("results/tabnet_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("TabNet ROC Curve")
plt.legend()
plt.savefig("results/tabnet_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(recall_curve, precision_curve, label=f"PR-AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("TabNet Precision-Recall Curve")
plt.legend()
plt.savefig("results/tabnet_pr_curve.png", dpi=300, bbox_inches='tight')
plt.show()


results_path = "results/tabnet_results.xlsx"

with pd.ExcelWriter(results_path, engine='openpyxl') as writer:

    # Sheet 1: Final Metrics
    metrics_excel_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        "Value": [accuracy, precision, recall, f1, roc_auc, pr_auc]
    })
    metrics_excel_df.to_excel(writer, sheet_name="Final_Metrics", index=False)

    # Sheet 2: Confusion Matrix
    cm_df = pd.DataFrame(cm,
                         columns=["Predicted_0", "Predicted_1"],
                         index=["Actual_0", "Actual_1"])
    cm_df.to_excel(writer, sheet_name="Confusion_Matrix")


    predictions_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Probability": y_pro
    })
    predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

print(f"\nAll results saved to: {results_path}")