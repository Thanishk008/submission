# code_thanishk — Execution Instructions

------------------------------------------------------------

## 1. Setup

```
python -m venv {name}
.\{name}\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------

## 2. Training

Trains the 1D CNN, saves the checkpoint to models/best_model.keras,
evaluates on the held-out test set, and saves a training history plot.
Run this first.
```
python train.py
```

Default parameters:
    --data_path    data/creditcard.csv
    --epochs       50
    --batch_size   32
    --lr           0.001
    --seed         42
    --out_dir      outputs/

------------------------------------------------------------

## 3. Testing (Optional)

Loads the saved checkpoint and evaluates on the test set.
This is optional since train.py already prints test-set metrics.
Useful for standalone verification of a saved model without retraining.
```
python test.py
```

Default parameters:
    --data_path    data/creditcard.csv
    --ckpt         models/best_model.keras

------------------------------------------------------------

## 4. Ablation Study

Runs Logistic Regression baseline + 3 ablation experiments
(SMOTE, Dropout, BatchNorm) and produces comparison tables + charts.
```
python run_ablation.py
```

Default parameters:
    --data_path    data/creditcard.csv
    --epochs       30
    --batch_size   32
    --lr           0.001
    --seed         42
    --out_dir      outputs/

------------------------------------------------------------

## 5. Foundation Model Comparison

Compares 1D CNN with pre-trained MobileNetV2 (ImageNet weights).
```
python run_foundation.py
```

Default parameters:
    --data_path    data/creditcard.csv
    --cnn_ckpt     models/best_model.keras
    --epochs       30
    --batch_size   32
    --lr           0.001
    --seed         42
    --out_dir      outputs/

------------------------------------------------------------

## 6. Error & Failure Analysis

Analyzes misclassifications (false positives / false negatives).
```
python run_error_analysis.py
```

Default parameters:
    --data_path    data/creditcard.csv
    --ckpt         models/best_model.keras
    --seed         42
    --out_dir      outputs/

------------------------------------------------------------

## 7. Unified Model Comparison

Produces a side-by-side comparison of all models (LR baseline, 1D CNN,
best ablation variant, MobileNetV2 foundation model) with grouped bar charts.

Best run after sections 4 and 5 to reuse cached JSON results.
If JSON files are not found, it falls back to training everything from scratch.
```
python run_comparison.py
```

Default parameters:
    --data_path    data/creditcard.csv
    --epochs       30
    --batch_size   32
    --lr           0.001
    --seed         42
    --out_dir      outputs/

Outputs: unified_comparison.csv, unified_comparison.png,
         unified_precision_recall.png

------------------------------------------------------------

## 8. Recommended Execution Order

```
python train.py              # 1. Train CNN + evaluate
python run_ablation.py        # 2. Baseline + ablation experiments
python run_foundation.py      # 3. MobileNetV2 vs CNN
python run_error_analysis.py  # 4. Misclassification analysis
python run_comparison.py      # 5. Unified comparison (uses cached JSON)
```

Note: test.py is optional — train.py already evaluates on the test set.
Run test.py only if you want to verify a saved checkpoint independently.

All scripts use default values. Parameters can be changed individually
via command-line arguments (see --help for each script).

------------------------------------------------------------

## 9. Submission Checklist

- [x] Dataset provided using Option A and placed correctly.
- [x] Model checkpoint instructions included.
- [x] requirements.txt generated and Python version specified.
- [x] Test command works.
- [x] Train command works.
- [x] Ablation study script included.
- [x] Foundation model comparison script included.
- [x] Error analysis script included.
- [x] Unified comparison script included.

------------------------------------------------------------