import streamlit as st
import numpy as np
import pandas as pd
import io

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

RANDOM_STATE = 42

FEATURES = [
    "income",
    "debt",
    "credit_history_years",
    "payment_history_score",
    "num_late_payments",
    "credit_utilization",
    "employment_years",
    "loan_amount",
]


st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide",
)


def generate_demo_dataset():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 2500

    data = pd.DataFrame({
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
        0.000018 * data["income"]
        - 0.000020 * data["debt"]
        + 0.12 * data["credit_history_years"]
        + 0.055 * data["payment_history_score"]
        - 0.42 * data["num_late_payments"]
        - 5.0 * data["credit_utilization"]
        + 0.06 * data["employment_years"]
        - 0.000012 * data["loan_amount"]
    )

    prob_good = 1 / (
        1 + np.exp(-(risk_score - risk_score.median()) / 2.2)
    )
    data["creditworthy"] = rng.binomial(1, prob_good)

    return data


def train_models(data):
    if "creditworthy" not in data.columns:
        raise ValueError(
            "The dataset must contain a 'creditworthy' target column."
        )

    missing = [col for col in FEATURES if col not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    work = data[FEATURES + ["creditworthy"]].copy()
    work["creditworthy"] = pd.to_numeric(
        work["creditworthy"], errors="coerce"
    )
    work = work.dropna(subset=["creditworthy"])

    unique_targets = set(work["creditworthy"].unique())
    if not unique_targets.issubset({0, 1}) or len(unique_targets) < 2:
        raise ValueError(
            "'creditworthy' must contain both 0 and 1 values."
        )

    X = work[FEATURES]
    y = work["creditworthy"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
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
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })

        fitted[name] = (pipe, pred, proba)

    results_df = pd.DataFrame(results).sort_values(
        "ROC-AUC", ascending=False
    ).reset_index(drop=True)

    return X_train, X_test, y_train, y_test, results_df, fitted


st.title("💳 Credit Scoring Model")
st.caption("CodeAlpha Task 1 • Machine Learning Creditworthiness Prediction")

st.sidebar.header("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls", "pdf"],
    help=(
        "Supported formats: CSV, Excel and PDF. "
        "The dataset must contain the required financial columns "
        "and a binary 'creditworthy' column with 0/1 values."
    ),
)

def read_uploaded_dataset(file):
    file_type = file.name.lower().split(".")[-1]

    if file_type == "csv":
        return pd.read_csv(file)

    if file_type in ["xlsx", "xls"]:
        return pd.read_excel(file)

    if file_type == "pdf":
        if pdfplumber is None:
            raise ValueError(
                "PDF support requires pdfplumber. Add pdfplumber "
                "to requirements.txt and redeploy."
            )

        tables = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_tables()
                for table in extracted:
                    if table and len(table) >= 2:
                        tables.append(table)

        if not tables:
            raise ValueError(
                "No table could be extracted from this PDF. "
                "Please upload a PDF containing a clear tabular dataset."
            )

        frames = []
        for table in tables:
            header = [str(x).strip() if x is not None else "" for x in table[0]]
            rows = table[1:]
            frames.append(pd.DataFrame(rows, columns=header))

        return pd.concat(frames, ignore_index=True)

    raise ValueError("Unsupported file format.")

if uploaded_file is not None:
    try:
        df = read_uploaded_dataset(uploaded_file)
        dataset_source = f"Uploaded {uploaded_file.name}"
        st.sidebar.success("Dataset uploaded successfully.")
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
else:
    df = generate_demo_dataset()
    dataset_source = "Built-in demonstration dataset"
    st.sidebar.info(
        "No file uploaded. Using the built-in demonstration dataset."
    )

try:
    X_train, X_test, y_train, y_test, results_df, fitted = train_models(df)
except ValueError as exc:
    st.error(str(exc))
    st.info(
        "Required columns: " + ", ".join(FEATURES) +
        ", plus the target column: creditworthy"
    )
    st.stop()

best_name = results_df.iloc[0]["Model"]

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Records", len(df))

with c2:
    st.metric("Features", len(FEATURES))

with c3:
    st.metric("Creditworthy", int(pd.to_numeric(
        df["creditworthy"], errors="coerce"
    ).sum()))

with c4:
    st.metric("Dataset Source", dataset_source)

with st.expander("🔎 Preview Dataset"):
    st.dataframe(df.head(100), use_container_width=True)

st.subheader("🤖 Model Comparison")

display_results = results_df.copy()
metric_columns = [
    "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
]
display_results[metric_columns] = display_results[metric_columns].round(4)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True,
)

selected_model = st.selectbox(
    "Choose a model for prediction",
    list(fitted.keys()),
    index=list(fitted.keys()).index(best_name),
)

st.info(f"Current model: **{selected_model}**")

st.subheader("🔮 Predict Creditworthiness")

left, right = st.columns(2)

with left:
    income = st.number_input(
        "Annual Income ($)",
        min_value=15000.0,
        max_value=180000.0,
        value=65000.0,
        step=1000.0,
    )

    debt = st.number_input(
        "Debt ($)",
        min_value=0.0,
        max_value=90000.0,
        value=22000.0,
        step=1000.0,
    )

    credit_history_years = st.number_input(
        "Credit History (Years)",
        min_value=0.5,
        max_value=25.0,
        value=8.0,
        step=0.5,
    )

    payment_history_score = st.slider(
        "Payment History Score",
        min_value=30,
        max_value=100,
        value=78,
    )

with right:
    num_late_payments = st.number_input(
        "Number of Late Payments",
        min_value=0,
        max_value=12,
        value=2,
        step=1,
    )

    credit_utilization = st.slider(
        "Credit Utilization",
        min_value=0.01,
        max_value=0.99,
        value=0.30,
        step=0.01,
    )

    employment_years = st.number_input(
        "Employment Years",
        min_value=0.0,
        max_value=30.0,
        value=6.0,
        step=0.5,
    )

    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=1000.0,
        max_value=70000.0,
        value=18000.0,
        step=1000.0,
    )

input_data = pd.DataFrame([{
    "income": income,
    "debt": debt,
    "credit_history_years": credit_history_years,
    "payment_history_score": payment_history_score,
    "num_late_payments": num_late_payments,
    "credit_utilization": credit_utilization,
    "employment_years": employment_years,
    "loan_amount": loan_amount,
}])

if st.button("🔍 Predict Creditworthiness", type="primary"):
    pipe, _, _ = fitted[selected_model]

    prediction = int(pipe.predict(input_data)[0])
    probability = float(pipe.predict_proba(input_data)[0][1])

    st.divider()

    result_col, prob_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.success("✅ Predicted: CREDITWORTHY")
        else:
            st.error("❌ Predicted: NOT CREDITWORTHY")

    with prob_col:
        st.metric(
            "Creditworthiness Probability",
            f"{probability * 100:.2f}%"
        )

st.subheader("📌 Model Evaluation")

pipe, pred, proba = fitted[selected_model]

cm = confusion_matrix(y_test, pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Not Creditworthy", "Actual: Creditworthy"],
    columns=["Predicted: Not Creditworthy", "Predicted: Creditworthy"],
)

st.write("### Confusion Matrix")
st.dataframe(cm_df, use_container_width=True)

st.write("### ROC Curve")
fpr, tpr, _ = roc_curve(y_test, proba)
roc_df = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr,
})
st.line_chart(roc_df.set_index("False Positive Rate"))

if selected_model in ["Decision Tree", "Random Forest"]:
    st.write("### Feature Importance")
    model = pipe.named_steps["model"]
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)
    st.bar_chart(importance)

st.subheader("💡 Business Interpretation")
st.write(
    """
- Higher income and stronger payment history generally improve
  creditworthiness.
- Higher debt, late payments, and credit utilization can increase risk.
- Loan amount and employment history are also considered by the model.
- ROC-AUC is used to compare the ranking performance of classifiers.
"""
)

st.caption(
    "For educational/demo purposes. Predictions are generated by a "
    "machine-learning model and should not be treated as financial advice."
            )
