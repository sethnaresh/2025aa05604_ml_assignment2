import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from model.evaluate import evaluate_models, plot_confusion_matrix


st.set_page_config(page_title="2025aa05604 ML Assignment 2 - Classification Models", layout="wide")

# Add logo to top-right corner
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("ML Assignment — Classification Models")
with col2:
    logo_path = Path("logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=240)

st.markdown("<h2 style='color: #1f77b4;'>2025aa05604 (Naresh Seth)</h2>", unsafe_allow_html=True)
st.caption("Upload your test dataset (CSV) to evaluate pre-trained classification models.")

with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write("All 6 classification models have been pre-trained on 80% of the dataset.")
    st.write("Upload a **test dataset (CSV)** with the same features to evaluate models.")

    st.header("📥 Download Test Dataset")
    st.write("Download the prepared test dataset (20% holdout set):")
    test_dataset_path = Path("data/test_dataset.csv")
    if test_dataset_path.exists():
        test_df = pd.read_csv(test_dataset_path)
        st.download_button(
            label="📥 Download test_dataset.csv",
            data=test_df.to_csv(index=False).encode("utf-8"),
            file_name="test_dataset.csv",
            mime="text/csv"
        )
    else:
        st.warning("Test dataset not found. Run training first.")

    st.header("1) Upload test dataset")
    st.write("Upload **a CSV file** with the same features as the training dataset.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    st.header("2) Model selection")
    model_name = st.selectbox(
        "Choose a model to inspect",
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"],
        index=0
    )


@st.cache_data(show_spinner=False)
def _load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_resource(show_spinner=True)
def _load_models_and_preprocessor():
    """Load pre-trained models and preprocessor from disk."""
    model_dir = Path("model/saved")
    
    if not model_dir.exists():
        return None, None, None, "No pre-trained models found. Please run: python train_offline.py --data <data.csv> --target <column>"
    
    try:
        # Load preprocessor
        with open(model_dir / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        
        # Load metadata
        with open(model_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # Load all models
        models = {}
        model_names = ["logreg", "dt", "knn", "nb", "rf", "xgb"]
        for name in model_names:
            model_path = model_dir / f"{name}_model.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    models[name] = pickle.load(f)
            else:
                models[name] = None
        
        return models, preprocessor, metadata, None
    except Exception as e:
        return None, None, None, f"Error loading models: {e}"


def main():
    # Load pre-trained models
    models, preprocessor, metadata, error_msg = _load_models_and_preprocessor()
    
    target_col = "Churned"  # Hardcoded target column name
    
    if error_msg:
        st.error(f"❌ {error_msg}")
        st.info("**Setup Instructions:**\n\n1. Run training offline:\n```bash\npython train_offline.py --data data/ecommerce_customer_churn_dataset.csv --target Churned\n```\n2. Then refresh this app")
        st.stop()
    
    if uploaded is None:
        st.info("⬅️ Upload a CSV test dataset from the sidebar to begin.")
        st.stop()

    df = _load_csv(uploaded)

    st.subheader("📊 Test Dataset Preview")
    st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    st.dataframe(df.head(6), use_container_width=True)

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the uploaded CSV.")
        st.stop()

    # Prepare test data
    try:
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        
        # Transform using pre-fitted preprocessor
        X_test_processed = preprocessor.transform(X_test)
    except Exception as e:
        st.error(f"Error preprocessing test data: {e}")
        st.stop()

    st.success("✅ Test data loaded and preprocessed successfully!")

    # Evaluate all models
    metrics_df, details = evaluate_models(models, X_test_processed, y_test)

    st.subheader("📈 Model Comparison (Evaluation Metrics)")
    st.dataframe(metrics_df, use_container_width=True)

    st.download_button(
        label="📥 Download metrics as CSV",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="model_metrics.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader(f"🔍 Detailed Analysis: {model_name}")

    model_key = {
        "Logistic Regression": "logreg",
        "Decision Tree": "dt",
        "kNN": "knn",
        "Naive Bayes": "nb",
        "Random Forest": "rf",
        "XGBoost": "xgb"
    }[model_name]

    m = models[model_key]
    
    if m is None:
        st.error(f"{model_name} model is not available (likely not installed).")
        st.stop()
    
    y_pred = m.predict(X_test_processed)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📊 Confusion Matrix")
        fig = plot_confusion_matrix(y_test, y_pred, title=f"{model_name} — Confusion Matrix")
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.markdown("### 📋 Classification Report")
        report_text = details[model_key]["classification_report"]
        st.code(report_text)


if __name__ == "__main__":
    main()
