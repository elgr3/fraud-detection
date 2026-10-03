import numpy as np
import pandas as pd

from fraud_detection import config
from fraud_detection.data import load_dataset, split3

COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _fake(n=2000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 30)), columns=COLUMNS)
    df[config.TARGET] = pd.Categorical(np.where(rng.random(n) < 0.05, "1", "0"))
    return df


def test_load_dataset_from_cache(tmp_path):
    _fake().to_parquet(tmp_path / "creditcard.parquet")
    X, y = load_dataset(cache_dir=tmp_path)
    assert list(X.columns) == COLUMNS
    assert set(y.unique()) <= {0, 1} and y.dtype == int


def test_split3_sizes_and_stratification(tmp_path):
    _fake().to_parquet(tmp_path / "creditcard.parquet")
    X, y = load_dataset(cache_dir=tmp_path)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split3(X, y)
    assert (len(X_tr), len(X_va), len(X_te)) == (1200, 400, 400)
    rates = [s.mean() for s in (y_tr, y_va, y_te)]
    assert max(rates) - min(rates) < 0.02
    assert not set(X_tr.index) & set(X_te.index)
