# README

------------------------------------------------------------

## 1. Project Overview

Project Title: Credit Card Fraud Detection using 1D CNN with SMOTE

Model Type:
1D Convolutional Neural Network (CNN)

Objective:
Binary Classification (Fraudulent vs Legitimate transactions)

Dataset Used:
Credit Card Fraud Detection — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Expected test evaluation for sanity check: F1-Score >= 0.750, ROC-AUC >= 0.95

------------------------------------------------------------

## 2. Repository Structure

```
submission/
  code_thanishk/
    data/
      readme.txt
    models/
      best_model.keras
    outputs/
    src/
      __init__.py
      baseline.py
      dataloader.py
      foundation_model.py
      model.py
      utils.py
    README.txt
    requirements.txt
    run_ablation.py
    run_comparison.py
    run_error_analysis.py
    run_foundation.py
    test.py
    train.py
  code_savitha/
    data/
    models/
    outputs/
    src/
      __init__.py
      baseline.py
      dataloader.py
      model.py
      utils.py
    README.txt
    requirements.txt
    test.py
    train.py
  presentation/
    presentation.mp4
    presentation.pptx
  reports/
    report_thanishk.pdf
    report_savitha.pdf
  README.txt
```

------------------------------------------------------------

## 3. Dataset (OPTION A — PUBLIC DATASET SPLITS)

Dataset Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset is automatically downloaded via kagglehub on first run.
Alternatively, download manually and place as data/creditcard.csv.

------------------------------------------------------------

## 4. Model Checkpoint

The best model checkpoint is automatically saved to models/best_model.keras
after running train.py.

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version:
3.9+

Framework: 
TensorFlow 2.20.0 (CPU-only execution)

How to install all dependencies:

1. Create a Python virtual environment:
```
python -m venv {name}
.\{name}\Scripts\activate
```

2. Install packages:
```
pip install -r requirements.txt
```

------------------------------------------------------------

## 6. Running the Code

Each team member's code directory contains its own README.txt
with execution instructions, default parameters, and run order.

- code_thanishk/README.txt
- code_savitha/README.txt

------------------------------------------------------------

## 7. Submission Checklist

- [x] Dataset provided using Option A and placed correctly.
- [x] Model checkpoint instructions included.
- [x] requirements.txt generated and Python version specified.
- [x] Test command works.
- [x] Train command works.
- [x] Ablation study script included.
- [x] Foundation model comparison script included.
- [x] Error analysis script included.
- [x] Unified comparison script included.
- [x] Code directories for both team members.
- [x] Report placeholders for both team members.

------------------------------------------------------------