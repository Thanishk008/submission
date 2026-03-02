import os
import zipfile
import subprocess
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from imblearn.over_sampling import SMOTE


SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


os.makedirs("results", exist_ok=True)


dataset_name = "mlg-ulb/creditcardfraud"

if not os.path.exists("creditcard.csv"):
    print("Downloading dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name])
    with zipfile.ZipFile("creditcardfraud.zip", "r") as zip_ref:
        zip_ref.extractall()
    print("Download complete.")

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

print("Dataset shape:", data.shape)
print("Fraud ratio:", sum(y) / len(y))


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


class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


train_dataset = FraudDataset(X_train_smote, y_train_smote)
val_dataset = FraudDataset(X_val_scaled, y_val)
test_dataset = FraudDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

class CNNModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(64 * 13, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model = CNNModel(dropout_rate=0.2).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 30
train_losses = []

val_metrics = {
    "epoch": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": []
}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)


    model.eval()
    val_preds = []
    val_probs = []
    val_labels_list = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            val_probs.extend(probs)
            val_preds.extend(preds)
            val_labels_list.extend(labels.numpy())

    val_acc = accuracy_score(val_labels_list, val_preds)
    val_prec = precision_score(val_labels_list, val_preds, zero_division=0)
    val_rec = recall_score(val_labels_list, val_preds)
    val_f1 = f1_score(val_labels_list, val_preds)
    val_roc_auc = roc_auc_score(val_labels_list, val_probs)

    val_metrics["epoch"].append(epoch + 1)
    val_metrics["accuracy"].append(val_acc)
    val_metrics["precision"].append(val_prec)
    val_metrics["recall"].append(val_rec)
    val_metrics["f1"].append(val_f1)
    val_metrics["roc_auc"].append(val_roc_auc)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")


plt.figure()
plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("CNN with SMOTE Training Loss")
plt.grid(True)
plt.savefig("results_cnn_smote_training_loss.png", dpi=300, bbox_inches='tight')
plt.show()


val_df = pd.DataFrame(val_metrics)
plt.figure(figsize=(10,6))
plt.plot(val_df["epoch"], val_df["accuracy"], marker='o', label="Accuracy")
plt.plot(val_df["epoch"], val_df["precision"], marker='x', label="Precision")
plt.plot(val_df["epoch"], val_df["recall"], marker='s', label="Recall")
plt.plot(val_df["epoch"], val_df["f1"], marker='^', label="F1 Score")
plt.plot(val_df["epoch"], val_df["roc_auc"], marker='d', label="ROC-AUC")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("CNN with SMOTE Validation Metrics")
plt.legend()
plt.grid(True)
plt.savefig("results_cnn_smote_validation_metrics.png", dpi=300, bbox_inches='tight')
plt.show()


model.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)

print("\n===== CNN with SMOTE Test Results =====")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)


plt.figure()
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("CNN with SMOTE Confusion Matrix")
plt.savefig("results/cnn_smote_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN with SMOTE ROC Curve")
plt.legend()
plt.savefig("results_cnn_smote_roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()


metrics = {
    "Model": ["CNN with SMOTE"],
    "Accuracy": [acc],
    "Precision": [prec],
    "Recall": [rec],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc],
    "Training_Loss_Final": [train_losses[-1]],
    "SMOTE": ["Applied"]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("results_cnn_smote_metrics.csv", index=False)
print("\nMetrics saved to results_cnn_smote_metrics.csv")