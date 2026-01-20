import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# XGBoost is optional; if not installed the app will still run without it.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def train_all_models(X_train, y_train, random_state: int = 42):
    """
    Train all 6 required classification models on the same training set.
    Returns a dict of fitted models.
    """
    models = {}

    models["logreg"] = LogisticRegression(max_iter=2000)
    models["dt"] = DecisionTreeClassifier(random_state=random_state)
    models["knn"] = KNeighborsClassifier(n_neighbors=5)
    models["nb"] = GaussianNB()
    models["rf"] = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )

    if _HAS_XGB:
        # for binary and multiclass classification
        models["xgb"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob" if len(set(y_train)) > 2 else "binary:logistic",
            eval_metric="logloss",
            random_state=random_state
        )
    else:
        models["xgb"] = None

    # Fit models
    for k, m in models.items():
        if m is None:
            continue
        m.fit(X_train, y_train)

    return models
