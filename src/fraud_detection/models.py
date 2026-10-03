from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection.config import SEED


@dataclass
class IsolationForestScorer:
    model: IsolationForest

    def score(self, X: pd.DataFrame) -> np.ndarray:
        # decision_function is high for normal points; flip so high = suspicious.
        return -self.model.decision_function(X)


def fit_isolation_forest(X_train: pd.DataFrame, y_train: pd.Series) -> IsolationForestScorer:
    model = IsolationForest(
        n_estimators=300,
        contamination=float(y_train.mean()),
        random_state=SEED,
        n_jobs=-1,
    )
    return IsolationForestScorer(model.fit(X_train))


def fit_smote_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=0.1, random_state=SEED)),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    eval_metric="aucpr",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipe.fit(X_train, y_train)
