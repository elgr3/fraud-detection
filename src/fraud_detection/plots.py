from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve  # noqa: E402


def plot_pr(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, s) in curves.items():
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    ax.legend()
    _save(fig, path)


def plot_confusion(y_true, flagged, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, flagged, display_labels=["legit", "fraud"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix at cost-optimal threshold")
    _save(fig, path)


def plot_cost(curves: dict, chosen: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in curves.items():
        ax.plot(curve["threshold"], curve["cost"], label=name)
        ax.axvline(chosen[name], ls="--", lw=1)
    ax.set(xlabel="Threshold", ylabel="Validation cost", title="Cost vs threshold", yscale="log")
    ax.legend()
    _save(fig, path)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
