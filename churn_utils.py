"""Shared data-cleaning and evaluation helpers for the Telco churn project.

Used by both the training notebook and the Streamlit app so that the
uploaded test CSV goes through exactly the same cleaning as the training data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def missing_columns(df, require_target=True):
    required = FEATURES + ([TARGET_COL] if require_target else [])
    return [c for c in required if c not in df.columns]


def clean_features(df):
    """Return the model-ready feature frame (raw categories, numeric dtypes fixed)."""
    X = df[FEATURES].copy()

    # TotalCharges ships as text; the blanks belong to tenure == 0 customers
    # who have not been billed yet, so 0 is the true value rather than a guess.
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    X.loc[X["TotalCharges"].isna() & (X["tenure"] == 0), "TotalCharges"] = 0.0

    X["SeniorCitizen"] = X["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna(
        X["SeniorCitizen"].astype(str)
    )
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).str.strip()
    return X


def encode_target(series):
    if series.dtype.kind in "biuf":
        return series.astype(int).to_numpy()
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0}).to_numpy()


def positive_scores(model, X):
    """Score for the positive class, used for AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def predict_at_threshold(model, X, threshold=0.5):
    # SVC.predict() uses the decision-function sign, which can disagree with its
    # Platt-scaled probabilities, so every model is labelled from the probability.
    return (positive_scores(model, X) >= threshold).astype(int)


def compute_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
