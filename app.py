import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from model.train_models import train_all_models
from model.evaluate import evaluate_models, plot_confusion_matrix
from model.utils import preprocess_train_test


st.set_page_config(page_title="2025aa05604 ML Assignment 2 - Classification Models", layout="wide")

st.title("2025aa05604 (Naresh Seth) Machine Learning Assignment 2 — Classification Models")
st.caption("Upload your dataset (CSV), train 6 classification models, compare metrics, and test predictions interactively.")

with st.sidebar:
    st.header("1) Upload dataset")
    st.write("Upload **a CSV classification dataset** (>=500 rows, >=12 features).")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    st.header("2) Target column")
    target_col = st.text_input("Enter target/label column name", value="target")

    st.header("3) Train/Test split")
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    st.header("4) Model selection")
    model_name = st.selectbox(
        "Choose a model",
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"],
        index=0
    )

    retrain = st.button("Train / Re-train models", type="primary")


@st.cache_data(show_spinner=False)
def _load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_resource(show_spinner=True)
def _train_cached(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test, preprocessor = preprocess_train_test(
        df=df, target_col=target, test_size=test_size, random_state=random_state
    )
    models = train_all_models(X_train, y_train, random_state=random_state)
    return X_train, X_test, y_train, y_test, preprocessor, models


def main():
    if uploaded is None:
        st.info("⬅️ Upload a CSV dataset from the sidebar to begin.")
        st.stop()

    df = _load_csv(uploaded)

    st.subheader("Dataset preview")
    st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    st.dataframe(df.head(20), use_container_width=True)

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the uploaded CSV.")
        st.stop()

    if retrain:
        st.cache_resource.clear()

    X_train, X_test, y_train, y_test, preprocessor, models = _train_cached(
        df=df, target=target_col, test_size=test_size, random_state=int(random_state)
    )

    st.success("✅ Models trained successfully!")

    # Evaluate all models
    metrics_df, details = evaluate_models(models, X_test, y_test)

    st.subheader("Model comparison (Evaluation metrics)")
    st.dataframe(metrics_df, use_container_width=True)

    st.download_button(
        label="Download metrics as CSV",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="model_metrics.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader(f"Selected model: {model_name}")

    model_key = {
        "Logistic Regression": "logreg",
        "Decision Tree": "dt",
        "kNN": "knn",
        "Naive Bayes": "nb",
        "Random Forest": "rf",
        "XGBoost": "xgb"
    }[model_name]

    m = models[model_key]
    y_pred = m.predict(X_test)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Confusion Matrix")
        fig = plot_confusion_matrix(y_test, y_pred, title=f"{model_name} — Confusion Matrix")
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.markdown("### Classification Report")
        report_text = details[model_key]["classification_report"]
        st.code(report_text)

    st.divider()
    st.subheader("Try single-row predictions")
    st.write("Provide feature values for a single row (JSON format), and get prediction + probability (when available).")

    feature_cols = [c for c in df.columns if c != target_col]
    example = {feature_cols[0]: df[feature_cols[0]].iloc[0]}
    json_input = st.text_area("Input features as JSON", value=str(example).replace("'", '"'), height=120)

    import json
    try:
        row = json.loads(json_input)
        row_df = pd.DataFrame([row], columns=feature_cols)
        # align and transform using the same preprocessing
        X_row = preprocessor.transform(row_df)

        pred = m.predict(X_row)[0]
        st.write("**Prediction:**", pred)

        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(X_row)[0]
            st.write("**Probabilities:**", probs)

    except Exception as e:
        st.warning(f"Invalid JSON or missing features: {e}")


if __name__ == "__main__":
    main()
