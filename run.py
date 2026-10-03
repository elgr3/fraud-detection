"""Train both detectors, tune thresholds on validation, report on test."""

import json

import numpy as np

from fraud_detection import config
from fraud_detection.data import load_dataset, split3
from fraud_detection.models import fit_isolation_forest, fit_smote_xgboost
from fraud_detection.plots import plot_confusion, plot_cost, plot_pr
from fraud_detection.threshold import choose_threshold, cost, metrics_at


def main() -> None:
    X, y = load_dataset()
    X_tr, X_va, X_te, y_tr, y_va, y_te = split3(X, y)

    iforest = fit_isolation_forest(X_tr, y_tr)
    xgb = fit_smote_xgboost(X_tr, y_tr)
    scorers = {
        "isolation_forest": iforest.score,
        "xgboost_smote": lambda d: xgb.predict_proba(d)[:, 1],
    }

    metrics, curves, chosen, test_scores = {}, {}, {}, {}
    for name, scorer in scorers.items():
        threshold, curve = choose_threshold(y_va, scorer(X_va))
        test_scores[name] = scorer(X_te)
        metrics[name] = metrics_at(y_te, test_scores[name], threshold)
        curves[name], chosen[name] = curve, threshold

    plot_pr({n: (y_te, s) for n, s in test_scores.items()}, config.FIGURES_DIR / "pr_curve.png")
    best = test_scores["xgboost_smote"] >= chosen["xgboost_smote"]
    plot_confusion(y_te, best, config.FIGURES_DIR / "confusion_matrix.png")
    plot_cost(curves, chosen, config.FIGURES_DIR / "cost_vs_threshold.png")

    report = {
        "dataset": {"n_rows": int(len(X)), "fraud_rate": round(float(y.mean()), 5)},
        "cost_assumption": {"c_fn": config.C_FN, "c_fp": config.C_FP},
        "baseline_flag_nothing_cost": cost(y_te, np.zeros(len(y_te), dtype=bool)),
        **metrics,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "metrics.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
