"""Cost maths and threshold selection."""

from __future__ import annotations

import numpy as np

from riskbeacon.config import CostAssumptions
from riskbeacon.evaluate import (
    approve_all_cost,
    best_threshold,
    expected_cost,
    ranking_metrics,
    savings_summary,
    threshold_metrics,
)

COSTS = CostAssumptions()


def test_cost_assumptions_derived():
    assert COSTS.cost_false_negative == 7000.0
    assert COSTS.cost_false_positive == 600.0


def test_expected_cost_hand_example():
    # Two clear defaults, two clear goods. At 0.5 the model is perfect here.
    y = [1, 1, 0, 0]
    p = [0.9, 0.8, 0.2, 0.1]
    assert expected_cost(y, p, 0.5, COSTS) == 0.0


def test_expected_cost_counts_errors():
    y = [1, 0]
    p = [0.1, 0.9]  # one false negative, one false positive
    assert expected_cost(y, p, 0.5, COSTS) == 7000.0 + 600.0


def test_approve_all_cost_is_all_defaults():
    y = np.array([1, 1, 0, 0, 1])
    assert approve_all_cost(y, COSTS) == 3 * 7000.0


def test_best_threshold_minimises_cost():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    t = best_threshold(y, p, COSTS)
    assert expected_cost(y, p, t, COSTS) == 0.0


def test_savings_summary_beats_baseline():
    y = [0, 0, 1, 1, 0, 1]
    p = [0.05, 0.1, 0.9, 0.85, 0.2, 0.7]
    summary = savings_summary(y, p, 0.5, COSTS)
    assert summary["dollars_saved"] > 0
    assert 0 <= summary["loss_reduction_pct"] <= 100


def test_ranking_and_threshold_metric_shapes():
    y = [0, 1, 0, 1]
    p = [0.2, 0.8, 0.3, 0.6]
    r = ranking_metrics(y, p)
    assert 0.0 <= r["roc_auc"] <= 1.0
    tm = threshold_metrics(y, p, 0.5)
    assert tm["true_positives"] + tm["false_negatives"] == 2
