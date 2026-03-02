import os
import zipfile
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random


SEED = 123
np.random.seed(SEED)
random.seed(SEED)

dataset_name = "mlg-ulb/creditcardfraud"

if not os.path.exists("creditcard.csv"):
    print("Downloading dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name])

    with zipfile.ZipFile("creditcardfraud.zip", "r") as zip_ref:
        zip_ref.extractall()

    print("Download  complete.")

data = pd.read_csv("creditcard.csv")

print("Dataset description:", data.shape)

X = data.drop("Class", axis=1)
y = data["Class"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

print("Train data size:", len(X_train))
print("Validation datasize:", len(X_val))
print("Test data size:", len(X_test))


print("\nTraining Logistic Regression ...")

model = LogisticRegression(
    #class_weight="balanced",
    max_iter=1000,
    random_state=SEED
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n Logistic Regression metrics ")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
