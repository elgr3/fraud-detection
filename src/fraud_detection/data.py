from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from fraud_detection.config import DATA_DIR, OPENML_ID, SEED, TARGET

CACHE_NAME = "creditcard.parquet"


def load_dataset(cache_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    cache = cache_dir / CACHE_NAME
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
        df.to_parquet(cache)
    y = df[TARGET].astype(str).astype(int).rename(TARGET)
    X = df.drop(columns=[TARGET]).astype(float)
    return X, y


def split3(X: pd.DataFrame, y: pd.Series):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
