# CodeAlpha Task 1 — Credit Scoring Model
# Predict creditworthiness from financial history.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# A reproducible demonstration dataset is generated because the CodeAlpha brief
# specifies example financial fields but does not provide a mandatory dataset.
rng = np.random.default_rng(RANDOM_STATE)
n = 2500

df = pd.DataFrame({
    "income": rng.normal(65000, 22000, n).clip(15000, 180000),
    "debt": rng.normal(22000, 12000, n).clip(0, 90000),
    "credit_history_years": rng.normal(8, 4, n).clip(0.5, 25),
    "payment_history_score": rng.normal(78, 13, n).clip(30, 100),
    "num_late_payments": rng.poisson(2.2, n).clip(0, 12),
    "credit_utilization": rng.beta(2.2, 4.5, n).clip(0.01, 0.99),
    "employment_years": rng.normal(6, 4, n).clip(0, 30),
    "loan_amount": rng.normal(18000, 10000, n).clip(1000, 70000),
})

# Higher score = better creditworthiness
risk_score = (
    0.000018 * df["income"]
    - 0.000020 * df["debt"]
    + 0.12 * df["credit_history_years"]
    + 0.055 * df["payment_history_score"]
    - 0.42 * df["num_late_payments"]
    - 5.0 * df["credit_utilization"]
    + 0.06 * df["employment_years"]
    - 0.000012 * df["loan_amount"]
)
prob_good = 1 / (1 + np.exp(-(risk_score - risk_score.median()) / 2.2))
df["creditworthy"] = rng.binomial(1, prob_good)

# Save dataset
df.to_csv("credit_scoring_dataset.csv", index=False)

X = df.drop(columns="creditworthy")
y = df["creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
}

results = []
fitted = {}

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba)
    })
    fitted[name] = (pipe, pred, proba)

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\nModel comparison:\n")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_pipe, best_pred, best_proba = fitted[best_name]

print(f"\nBest model: {best_name}\n")
print(classification_report(y_test, best_pred, target_names=["Not Creditworthy", "Creditworthy"]))

ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, display_labels=["Not Creditworthy", "Creditworthy"]
)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(y_test, best_proba)
plt.title(f"ROC Curve — {best_name}")
plt.tight_layout()
plt.show()

# Feature importance when the best model is tree-based
if best_name in ["Decision Tree", "Random Forest"]:
    model = best_pipe.named_steps["model"]
    importance = pd.Series(model.feature_importances_, index=numeric_features)
    importance.sort_values(ascending=False).head(10).sort_values().plot.barh()
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

print("\nBusiness interpretation:")
print("- Higher income and stronger payment history generally improve creditworthiness.")
print("- Higher debt, late payments, utilization, and large loan amounts increase risk.")
print("- ROC-AUC is useful for comparing ranking quality across classifiers.")
