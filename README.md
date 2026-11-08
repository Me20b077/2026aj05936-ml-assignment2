# Telco Customer Churn — Classification Model Comparison & Streamlit App

**ML Assignment 2 · M.Tech (AIML) · Student ID: 2026aj05936**

- **GitHub repository:** https://github.com/Me20b077/2026aj05936-ml-assignment2
- **Live Streamlit app:** https://2026aj05936-ml-assignment2-8akbji8qjup3afytpwe7t2.streamlit.app/

---

## a. Problem statement

A telecom operator loses revenue every time a subscriber cancels their service (**churn**). Keeping an existing customer is much cheaper than acquiring a new one, so the operator wants to identify customers who are likely to churn **before** they leave and target them with retention offers.

The task is a **binary classification** problem: given a customer's demographics, subscribed services, contract and billing details, predict `Churn` (**Yes / No**). Five classification models — Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes and Support Vector Machine — are trained on the same data, compared on Accuracy, AUC, Precision, Recall, F1 and MCC, and deployed in an interactive Streamlit app.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Telco Customer Churn (IBM sample dataset) |
| Source | Kaggle — <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> |
| Instances | 7,043 customers |
| Features | 19 input features (+ `customerID` identifier, dropped) |
| Target | `Churn` — Yes (1,869 · 26.5 %) / No (5,174 · 73.5 %) → imbalanced binary target |
| Train / test split | 80 / 20 stratified → 5,634 train rows, 1,409 test rows (`test_data.csv`) |

**Features**

| Group | Columns | Type |
|---|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | categorical (binary) |
| Account | `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod` | numeric / categorical |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical |
| Billing | `MonthlyCharges`, `TotalCharges` | numeric |

**Data quality & EDA highlights** (full analysis in `model_training.ipynb`)
- No duplicate rows/IDs and no `NaN`s, but `TotalCharges` is stored as text with **11 blank values** — all for customers with `tenure = 0` (never billed). They are converted to numeric and filled with `0`.
- No outliers by the 1.5 × IQR rule; numeric features are on very different scales → standardised.
- Strongest churn drivers: **month-to-month contract** (43 % churn vs 3 % for two-year), **short tenure** (median 10 vs 38 months), **fiber-optic internet** (42 %), **electronic-check payment** (45 %) and **no online-security / tech-support** add-ons. `gender` and `PhoneService` are almost uninformative.
- `TotalCharges` ≈ `tenure × MonthlyCharges` (r = 0.9996) → multicollinearity, handled by regularisation / non-linear models.

**Pre-processing pipeline** (saved inside every model file): median imputation + `StandardScaler` for the 3 numeric features, `OneHotEncoder(drop="if_binary", handle_unknown="ignore")` for the 16 categorical features → 40 model inputs.

## c. GitHub repository link

https://github.com/Me20b077/2026aj05936-ml-assignment2

```
2026aj05936-ml-assignment2/
├── app.py                    # Streamlit app (loads saved models, no retraining)
├── churn_utils.py            # shared cleaning + metric helpers (notebook & app)
├── model_training.ipynb      # EDA, preprocessing, tuning, evaluation, model export
├── requirements.txt
├── README.md
├── test_data.csv             # held-out test split (1,409 rows, includes Churn)
├── data/
│   └── telco_customer_churn.csv   # original Kaggle dataset
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metrics.json          # tuned hyper-parameters + test metrics
└── .streamlit/config.toml    # app theme
```

## d. Models used

All five models were trained on the same 80 % training split. Hyper-parameters were tuned with `GridSearchCV` (5-fold stratified CV, **F1 of the churn class** as the selection metric, because the target is imbalanced and accuracy rewards predicting "No"). Metrics below are on the **held-out test set (1,409 rows)**; a customer is labelled as a churner when the predicted churn probability ≥ 0.5.

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | C = 0.3, class_weight = balanced |
| Decision Tree | max_depth = 3, min_samples_leaf = 1, criterion = gini, class_weight = balanced |
| KNN | n_neighbors = 61, weights = uniform, p = 1 (Manhattan) |
| Naive Bayes (Gaussian) | var_smoothing = 1.0 |
| Support Vector Machine | RBF kernel, C = 0.3, gamma = 0.1, class_weight = balanced, probability = True |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7622 | **0.8486** | 0.5356 | 0.7834 | **0.6363** | 0.4875 |
| Decision Tree | 0.7480 | 0.8202 | 0.5171 | 0.7701 | 0.6187 | 0.4607 |
| KNN | 0.7928 | 0.8429 | 0.6178 | 0.5749 | 0.5956 | 0.4570 |
| Naive Bayes | 0.7523 | 0.8327 | 0.5217 | **0.8021** | 0.6322 | 0.4820 |
| Support Vector Machine | **0.7991** | 0.8324 | **0.6194** | 0.6310 | 0.6252 | **0.4880** |

*(Bold = best value in each column. A majority-class baseline scores Accuracy 0.735 but F1 = MCC = 0.)*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best **AUC (0.849)** and **F1 (0.636)**, and an MCC (0.4875) practically tied with the best. With balanced class weights and mild L2 regularisation it finds 78 % of churners. Its strength shows that, after one-hot encoding, churn risk is close to **linearly separable** (contract, tenure, fiber, e-check and missing add-ons each add risk). It was also the best model in cross-validation (CV F1 0.630), is insensitive to C, fast and fully interpretable. |
| Decision Tree | Untuned, the tree overfit (CV F1 ≈ 0.50, AUC ≈ 0.66). Limiting it to **depth 3** with balanced weights raises F1 to 0.619 and recall to 0.77, but it has the **lowest AUC (0.820)**: with only 8 leaves it outputs a few distinct probabilities and ranks customers coarsely. Its rules are very readable — month-to-month customers on fiber optic, or new month-to-month customers, are flagged as churners. |
| KNN | Good accuracy (0.793) and precision (0.618) but the **lowest recall (0.575)**, F1 and MCC. KNN cannot use class weights, so with 73 % non-churners in each neighbourhood it predicts "No" too often and misses ~4 in 10 churners. A large k (61) was needed for stability; it is also the slowest model at prediction time. |
| Naive Bayes | Highest **recall (0.802)** and a strong F1 (0.632) despite violating the feature-independence assumption (`TotalCharges` ≈ `tenure × MonthlyCharges`, repeated "No internet service" levels); heavy `var_smoothing` was needed to temper over-confidence. Its test F1 is 0.025 above its CV F1, so part of this result is split luck — in CV it ranks below LR and SVM. |
| Support Vector Machine | Best **accuracy (0.799)**, **precision (0.619)** and **MCC (0.488)** with moderate recall (0.631). Platt probability calibration pulls predictions toward the true 26.5 % churn rate, making it more conservative than LR (fewer false alarms, more missed churners). Its AUC (0.832) is below LR's, so the RBF kernel gives no ranking gain over the linear model; it is also the most expensive to train. |
| **Overall, winner for this dataset** | **Logistic Regression** — best AUC and F1, MCC tied with SVM (Δ = 0.0005), best in cross-validation, and the simplest and most explainable model. For a retention campaign its higher recall (78 % vs 63 % of churners found) outweighs the lower precision. SVM is the runner-up and preferable only if false alarms are costly. |

## Streamlit app

Features:
1. **CSV upload** of test data (or one-click use of the bundled `test_data.csv`, which can also be downloaded from the sidebar)
2. **Model selection dropdown** for the five saved models, plus an adjustable decision threshold
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC
4. **Confusion matrix and classification report**
5. Side-by-side comparison of all five models (table, bar chart, ROC curves), a downloadable predictions table and a data preview

The uploaded CSV must contain the 19 feature columns (same names as the original dataset) and the `Churn` column for evaluation. Without `Churn`, the app still shows predictions. Models are loaded from `model/` — nothing is retrained in the app.

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
streamlit run app.py

# to re-run the training notebook (re-creates model/ and test_data.csv)
pip install jupyter ipykernel
jupyter notebook model_training.ipynb
```

**Deployment (Streamlit Community Cloud):** New app → select this repository → branch `main` → main file `app.py` → *Advanced settings* → Python **3.12** → Deploy.
