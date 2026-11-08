import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from churn_utils import (
    ID_COL,
    MODEL_FILES,
    TARGET_COL,
    clean_features,
    compute_metrics,
    encode_target,
    missing_columns,
    positive_scores,
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_TEST_FILE = BASE_DIR / "test_data.csv"
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Telco Churn Classifier Lab", page_icon="📉", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_training_info():
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def read_csv(file):
    return pd.read_csv(file)


def score_all(models, X):
    return {name: positive_scores(model, X) for name, model in models.items()}


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum()
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, annot_kws={"fontsize": 12},
                xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc(y_true, scores, highlight):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, lw=2.5 if name == highlight else 1, alpha=1 if name == highlight else 0.5, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("**1. Test data**")
uploaded = st.sidebar.file_uploader("Upload a test CSV", type=["csv"],
                                    help="Raw Telco churn columns, including the `Churn` target column.")
use_sample = st.sidebar.toggle("Use bundled test_data.csv instead", value=uploaded is None)
if SAMPLE_TEST_FILE.exists():
    st.sidebar.download_button("⬇️ Download sample test_data.csv", SAMPLE_TEST_FILE.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

st.sidebar.markdown("**2. Model**")
model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES))
threshold = st.sidebar.slider("Decision threshold (churn probability)", 0.05, 0.95, 0.50, 0.05,
                              help="Customers with a predicted churn probability ≥ threshold are flagged as churners. "
                                   "AUC does not depend on the threshold; the other metrics do.")

info = load_training_info()
best_params = info.get("best_params", {}).get(model_name)
if best_params:
    st.sidebar.markdown("**Tuned hyper-parameters**")
    st.sidebar.json(best_params, expanded=False)

# ---------------------------------------------------------------- header
st.title("📉 Telco Customer Churn — Classifier Lab")
st.caption("ML Assignment 2 · 2026aj05936 · Five pre-trained models scored on your test data (no retraining in the app).")

if uploaded is not None and not use_sample:
    raw = read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_TEST_FILE.exists():
    raw = read_csv(SAMPLE_TEST_FILE)
    source = "bundled test_data.csv"
else:
    st.info("Upload a test CSV in the sidebar or switch on **Use bundled test_data.csv** to begin.")
    st.stop()

missing_feats = missing_columns(raw, require_target=False)
if missing_feats:
    st.error(f"The uploaded file is missing required feature columns: {', '.join(missing_feats)}")
    st.stop()

has_target = TARGET_COL in raw.columns
X = clean_features(raw)
models = load_models()
all_scores = score_all(models, X)
scores = all_scores[model_name]
y_pred = (scores >= threshold).astype(int)

st.markdown(f"**Data:** `{source}` &nbsp;·&nbsp; **Rows scored:** {len(raw):,} &nbsp;·&nbsp; "
            f"**Flagged as churners by {model_name}:** {y_pred.sum():,} ({y_pred.mean():.1%})")

if not has_target:
    st.warning(f"No `{TARGET_COL}` column found — showing predictions only. "
               "Include the target column to see evaluation metrics.")

tabs = st.tabs(["📊 Model evaluation", "🏁 Compare all models", "🔮 Predictions", "🗂️ Data preview"])

# ---------------------------------------------------------------- evaluation
with tabs[0]:
    if has_target:
        y_true = encode_target(raw[TARGET_COL])
        valid = ~pd.isna(y_true)
        y_true, y_eval, s_eval = y_true[valid].astype(int), y_pred[valid], scores[valid]
        m = compute_metrics(y_true, y_eval, s_eval)

        st.subheader(f"{model_name} · threshold {threshold:.2f}")
        for row in (METRIC_ORDER[:3], METRIC_ORDER[3:]):
            for col, key in zip(st.columns(3, border=True), row):
                col.metric(key, f"{m[key]:.4f}")
        reference = info.get("test_metrics", {}).get(model_name, {})
        if source == "bundled test_data.csv" and threshold == 0.5 and reference:
            same = all(abs(m[k] - reference[k]) < 1e-4 for k in METRIC_ORDER)
            st.caption("✅ Matches the test-set metrics reported in the notebook / README." if same
                       else "⚠️ Differs from the notebook metrics — check that the model files are up to date.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(plot_confusion(y_true, y_eval), width="content")
        with right:
            st.markdown("**Classification report**")
            rep = classification_report(y_true, y_eval, target_names=["No churn", "Churn"],
                                        output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T.drop(index="accuracy")
            st.dataframe(rep_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                              "f1-score": "{:.4f}", "support": "{:.0f}"}),
                         width="stretch")
            tn, fp, fn, tp = confusion_matrix(y_true, y_eval, labels=[0, 1]).ravel()
            st.markdown(f"- **{tp}** churners correctly flagged, **{fn}** churners missed\n"
                        f"- **{fp}** loyal customers flagged unnecessarily (false alarms)")
    else:
        st.info("Evaluation needs the `Churn` column in the uploaded file.")

# ---------------------------------------------------------------- comparison
with tabs[1]:
    if has_target:
        y_true = encode_target(raw[TARGET_COL])
        valid = ~pd.isna(y_true)
        y_true = y_true[valid].astype(int)
        rows = {name: compute_metrics(y_true, (s[valid] >= threshold).astype(int), s[valid])
                for name, s in all_scores.items()}
        comp = pd.DataFrame(rows).T[METRIC_ORDER]
        comp.index.name = "ML Model Name"

        st.subheader(f"All five models on {source} @ threshold {threshold:.2f}")
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"), width="stretch")
        best = comp["F1"].idxmax()
        st.success(f"Best F1 on this data: **{best}** (F1 = {comp.loc[best, 'F1']:.4f}, "
                   f"MCC = {comp.loc[best, 'MCC']:.4f})")

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6, 4))
            comp.plot(kind="bar", ax=ax, width=0.8)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            ax.legend(ncol=3, fontsize=7, loc="lower right")
            ax.set_xlabel("")
            fig.tight_layout()
            st.pyplot(fig)
        with right:
            st.pyplot(plot_roc(y_true, {n: s[valid] for n, s in all_scores.items()}, model_name))
    else:
        st.info("Model comparison needs the `Churn` column in the uploaded file.")

# ---------------------------------------------------------------- predictions
with tabs[2]:
    out = pd.DataFrame({
        ID_COL: raw[ID_COL] if ID_COL in raw.columns else np.arange(len(raw)),
        "churn_probability": np.round(scores, 4),
        "predicted_churn": np.where(y_pred == 1, "Yes", "No"),
    })
    if has_target:
        out["actual_churn"] = raw[TARGET_COL].values
        out["correct"] = out["predicted_churn"] == out["actual_churn"]
    only_risky = st.checkbox("Show only customers flagged as churners", value=False)
    view = out[out["predicted_churn"] == "Yes"] if only_risky else out
    st.dataframe(view.sort_values("churn_probability", ascending=False), width="stretch", height=420)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv", mime="text/csv")

# ---------------------------------------------------------------- data preview
with tabs[3]:
    st.dataframe(raw.head(50), width="stretch")
    if has_target:
        left, right = st.columns(2)
        with left:
            st.markdown("**Churn distribution in uploaded data**")
            st.bar_chart(raw[TARGET_COL].value_counts())
        with right:
            st.markdown("**Churn rate by contract type**")
            rate = raw.assign(_y=encode_target(raw[TARGET_COL])).groupby("Contract")["_y"].mean()
            st.bar_chart(rate)

with st.expander("ℹ️ About the models"):
    st.markdown(
        "All five models are scikit-learn pipelines (median imputation + standard scaling for numeric features, "
        "one-hot encoding for categorical features, then the classifier). They were trained and tuned with "
        "5-fold stratified cross-validation (F1 objective) in `model_training.ipynb` and saved in `model/`. "
        "The app only loads these files and calls `predict_proba`; nothing is retrained on uploaded data."
    )
