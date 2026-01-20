import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report
)


def _safe_auc(y_true, y_proba, average="weighted"):
    """
    Computes AUC safely for binary or multi-class.
    Returns np.nan if proba not available.
    """
    try:
        # binary
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            return roc_auc_score(y_true, y_proba)
        # multiclass
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
    except Exception:
        return np.nan


def evaluate_models(models: dict, X_test, y_test):
    rows = []
    details = {}

    is_multiclass = len(np.unique(y_test)) > 2

    for key, model in models.items():
        if model is None:
            rows.append({
                "ML Model Name": "XGBoost",
                "Accuracy": np.nan, "AUC": np.nan, "Precision": np.nan,
                "Recall": np.nan, "F1": np.nan, "MCC": np.nan
            })
            details[key] = {"classification_report": "XGBoost not installed. Add xgboost in requirements.txt."}
            continue

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if not is_multiclass:
                # use positive class proba for binary
                y_auc_input = y_proba[:, 1]
            else:
                y_auc_input = y_proba
        else:
            y_auc_input = None

        acc = accuracy_score(y_test, y_pred)

        avg = "weighted" if is_multiclass else "binary"
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

        auc = _safe_auc(y_test, y_auc_input) if y_auc_input is not None else np.nan
        mcc = matthews_corrcoef(y_test, y_pred)

        name = {
            "logreg": "Logistic Regression",
            "dt": "Decision Tree",
            "knn": "kNN",
            "nb": "Naive Bayes",
            "rf": "Random Forest",
            "xgb": "XGBoost"
        }.get(key, key)

        rows.append({
            "ML Model Name": name,
            "Accuracy": round(float(acc), 4),
            "AUC": round(float(auc), 4) if not np.isnan(auc) else np.nan,
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
            "MCC": round(float(mcc), 4),
        })

        details[key] = {
            "classification_report": classification_report(y_test, y_pred, digits=4)
        }

    metrics_df = pd.DataFrame(rows)
    # Sort in assignment order
    order = ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
    metrics_df["__order"] = metrics_df["ML Model Name"].apply(lambda x: order.index(x) if x in order else 999)
    metrics_df = metrics_df.sort_values("__order").drop(columns="__order")

    return metrics_df, details


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # annotate
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")

    fig.tight_layout()
    return fig
