import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Scoring Model")
st.write(
    "Machine Learning based creditworthiness prediction — "
    "CodeAlpha Task 1."
)

# -----------------------------
# Generate Dataset
# -----------------------------
@st.cache_data
def generate_dataset():
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

    prob_good = 1 / (
        1 + np.exp(-(risk_score - risk_score.median()) / 2.2)
    )

    df["creditworthy"] = rng.binomial(1, prob_good)

    return df


# -----------------------------
# Train Models
# -----------------------------
@st.cache_resource
def train_models(df):

    X = df.drop(columns="creditworthy")
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    results = []
    fitted = {}

    for name, model in models.items():

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1": f1_score(
                y_test, pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, proba)
        })

        fitted[name] = (pipe, pred, proba)

    results_df = pd.DataFrame(results).sort_values(
        "ROC-AUC",
        ascending=False
    )

    best_name = results_df.iloc[0]["Model"]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        results_df,
        fitted,
        best_name
    )


df = generate_dataset()

X_train, X_test, y_train, y_test, results_df, fitted, best_name = train_models(df)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📊 Model Selection")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(fitted.keys()),
    index=list(fitted.keys()).index(best_name)
)

# -----------------------------
# Dataset
# -----------------------------
st.subheader("📁 Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Features", len(df.columns) - 1)

with col3:
    st.metric(
        "Creditworthy %",
        f"{df['creditworthy'].mean() * 100:.1f}%"
    )

with st.expander("View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)


# -----------------------------
# Model Comparison
# -----------------------------
st.subheader("🤖 Model Comparison")

display_results = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]:
    display_results[column] = display_results[column].round(4)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)

st.success(f"Selected Best Model: {best_name}")


# -----------------------------
# Prediction Form
# -----------------------------
st.subheader("🔮 Creditworthiness Prediction")

st.write("Enter applicant financial information:")

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "Annual Income ($)",
        min_value=15000.0,
        max_value=180000.0,
        value=65000.0,
        step=1000.0
    )

    debt = st.number_input(
        "Debt ($)",
        min_value=0.0,
        max_value=90000.0,
        value=22000.0,
        step=1000.0
    )

    credit_history_years = st.number_input(
        "Credit History (Years)",
        min_value=0.5,
        max_value=25.0,
        value=8.0,
        step=0.5
    )

    payment_history_score = st.slider(
        "Payment History Score",
        min_value=30,
        max_value=100,
        value=78
    )

with col2:

    num_late_payments = st.number_input(
        "Number of Late Payments",
        min_value=0,
        max_value=12,
        value=2
    )

    credit_utilization = st.slider(
        "Credit Utilization",
        min_value=0.01,
        max_value=0.99,
        value=0.30,
        step=0.01
    )

    employment_years = st.number_input(
        "Employment Years",
        min_value=0.0,
        max_value=30.0,
        value=6.0,
        step=0.5
    )

    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=1000.0,
        max_value=70000.0,
        value=18000.0,
        step=1000.0
    )


input_data = pd.DataFrame({
    "income": [income],
    "debt": [debt],
    "credit_history_years": [credit_history_years],
    "payment_history_score": [payment_history_score],
    "num_late_payments": [num_late_payments],
    "credit_utilization": [credit_utilization],
    "employment_years": [employment_years],
    "loan_amount": [loan_amount]
})


if st.button("🔍 Predict Creditworthiness", type="primary"):

    pipe, _, _ = fitted[selected_model]

    prediction = pipe.predict(input_data)[0]
    probability = pipe.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.success("✅ Applicant is predicted to be CREDITWORTHY")
    else:
        st.error("❌ Applicant is predicted to be NOT CREDITWORTHY")

    st.metric(
        "Creditworthiness Probability",
        f"{probability * 100:.2f}%"
    )


# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📌 Model Evaluation")

pipe, pred, proba = fitted[selected_model]

cm = confusion_matrix(y_test, pred)

st.write("### Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not Creditworthy", "Actual: Creditworthy"],
    columns=["Predicted: Not Creditworthy", "Predicted: Creditworthy"]
)

st.dataframe(cm_df, use_container_width=True)


# -----------------------------
# ROC Curve
# -----------------------------
st.write("### ROC Curve")

fpr, tpr, _ = roc_curve(y_test, proba)

roc_df = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr
})

st.line_chart(
    roc_df.set_index("False Positive Rate")
)


# -----------------------------
# Feature Importance
# -----------------------------
if selected_model in ["Decision Tree", "Random Forest"]:

    st.subheader("📈 Feature Importance")

    model = pipe.named_steps["model"]

    features = X_train.columns

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    st.bar_chart(importance)


# -----------------------------
# Business Interpretation
# -----------------------------
st.subheader("💡 Business Interpretation")

st.write("""
- Higher income and stronger payment history generally improve creditworthiness.
- Higher debt, late payments and credit utilization can increase credit risk.
- Loan amount and employment history are also considered by the model.
- ROC-AUC is used to compare the ranking performance of different classifiers.
""")

st.caption(
    "CodeAlpha Task 1 — Credit Scoring Model | "
    "Machine Learning Project"
)
