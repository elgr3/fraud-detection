import numpy as np
import pytest

from fraud_detection.threshold import choose_threshold, cost, metrics_at


def test_cost_counts_missed_frauds_and_false_alerts():
    y = np.array([1, 1, 0, 0])
    flagged = np.array([True, False, True, False])
    assert cost(y, flagged, c_fn=100, c_fp=1) == 100 + 1


def test_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t, curve = choose_threshold(y, s, c_fn=100, c_fp=1)
    assert 0.3 < t <= 0.8
    assert curve["cost"].min() == 0


def test_high_fn_cost_pushes_threshold_down():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = y * 0.3 + rng.random(2000)
    t_strict, _ = choose_threshold(y, s, c_fn=1, c_fp=1)
    t_lenient, _ = choose_threshold(y, s, c_fn=50, c_fp=1)
    assert t_lenient < t_strict


def test_constant_scores_still_return_finite_threshold():
    t, curve = choose_threshold(np.array([0, 1, 0]), np.zeros(3))
    assert np.isfinite(t) and np.isfinite(curve["cost"]).all()


def test_metrics_when_nothing_flagged_are_zero_not_errors():
    m = metrics_at(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.1, 0.2]), threshold=0.9)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["n_flagged"] == 0
    assert m["cost"] == pytest.approx(200.0)
