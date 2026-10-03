import numpy as np
import pandas as pd

from fraud_detection.plots import plot_confusion, plot_cost, plot_pr


def test_plots_write_files(tmp_path):
    y = np.array([0, 1, 0, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.4])
    plot_pr({"m": (y, s)}, tmp_path / "pr.png")
    plot_confusion(y, s >= 0.5, tmp_path / "cm.png")
    curve = pd.DataFrame({"threshold": [0.1, 0.5, 0.9], "cost": [4.0, 0.0, 200.0]})
    plot_cost({"m": curve}, {"m": 0.5}, tmp_path / "cost.png")
    for name in ("pr.png", "cm.png", "cost.png"):
        assert (tmp_path / name).stat().st_size > 0
