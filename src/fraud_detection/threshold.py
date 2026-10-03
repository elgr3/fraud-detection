import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from fraud_detection.config import C_FN, C_FP


def cost(y_true, flagged, c_fn: float = C_FN, c_fp: float = C_FP) -> float:
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    missed = np.sum(y_true & ~flagged)
    false_alerts = np.sum(~y_true & flagged)
    return float(c_fn * missed + c_fp * false_alerts)


def choose_threshold(y_true, scores, c_fn=C_FN, c_fp=C_FP, n_grid: int = 1000):
    scores = np.asarray(scores, dtype=float)
    grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_grid + 1)))
    grid = np.append(grid, np.nextafter(scores.max(), np.inf))  # "flag nothing" option
    costs = np.array([cost(y_true, scores >= t, c_fn, c_fp) for t in grid])
    curve = pd.DataFrame({"threshold": grid, "cost": costs})
    return float(grid[int(np.argmin(costs))]), curve


def metrics_at(y_true, scores, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold
    return {
        "precision": round(float(precision_score(y_true, flagged, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, flagged, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, flagged, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "cost": cost(y_true, flagged),
        "n_flagged": int(flagged.sum()),
        "threshold": round(float(threshold), 6),
    }
