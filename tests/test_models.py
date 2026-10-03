import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_detection.models import fit_isolation_forest, fit_smote_xgboost


def _data(n=3000):
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.03).astype(int)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    X.loc[y == 1, ["a", "b"]] += 4  # frauds are outliers
    return X, pd.Series(y)


def test_isolation_forest_scores_frauds_higher():
    X, y = _data()
    scores = fit_isolation_forest(X, y).score(X)
    assert roc_auc_score(y, scores) > 0.9


def test_smote_xgboost_learns_and_does_not_resample_at_predict():
    X, y = _data()
    pipe = fit_smote_xgboost(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert roc_auc_score(y, proba) > 0.95
