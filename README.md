# Predicting Credit Card Approvals

This project builds an automatic credit card approval predictor using **Logistic Regression** with a full preprocessing and model-selection workflow in Jupyter Notebook.

The notebook demonstrates:
- handling missing values
- processing categorical features
- scaling numeric features
- dealing with unbalanced data (`class_weight` + imbalance-aware metrics)
- automatic hyperparameter optimization with `GridSearchCV`

## Files

- `Predicting_Credit_Card_Approvals.ipynb`: end-to-end guided notebook
- `models/`: saved trained model pipeline (created when notebook runs)

## Quick Start

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter
```

3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open and run:

`Predicting_Credit_Card_Approvals.ipynb`

## Dataset

By default, the notebook loads the UCI Credit Approval dataset from:

`https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data`

If you prefer a local file, place it at:

`data/credit_approval.csv`

The notebook auto-detects this local path first.

## Workflow in the Notebook

1. Load data and standardize column names.
2. Replace placeholder missing values (`?`) with `NaN`.
3. Normalize target labels to binary (`0/1`).
4. Detect numeric vs categorical columns.
5. Build a preprocessing pipeline:
   - numeric: `SimpleImputer(strategy="median")` + `StandardScaler`
   - categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder`
6. Split data with stratification.
7. Train Logistic Regression with `GridSearchCV`.
8. Include imbalance handling in tuning (`class_weight=None` vs `"balanced"`).
9. Evaluate with metrics including `balanced_accuracy`, `f1`, and `roc_auc`.
10. Save the best model pipeline as a `.joblib` artifact.

## Adapting to LendingClub (Loan Approval Use Case)

You can replicate the same pipeline on LendingClub (or similar loan datasets):
- replace the data-loading cell with your loan dataset
- map your approval/default target to binary `0/1`
- keep the same preprocessing + GridSearchCV structure
- adjust metrics to your business objective (for example, prioritize recall for risky loans)
