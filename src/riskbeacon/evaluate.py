"""Scoring metrics and the dollars-and-cents view of a threshold.

Ranking metrics (ROC AUC, average precision) tell me how well the model orders
borrowers by risk. They do not tell me where to draw the approve / decline line.
For that I turn the confusion matrix into money using the cost assumptions and
pick the threshold that minimises expected loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CostAssumptions


def ranking_metrics(y_true, y_prob) -> dict[str, float]:
    """Threshold-free quality of the risk ranking."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def threshold_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 and the confusion counts at one threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def expected_cost(y_true, y_prob, threshold: float, costs: CostAssumptions) -> float:
    """Total dollar cost of decisions at a threshold.

    A false negative is a default we approved; a false positive is a good
    customer we turned away.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp * costs.cost_false_positive + fn * costs.cost_false_negative


@dataclass(frozen=True)
class CostCurvePoint:
    threshold: float
    cost: float


def cost_curve(y_true, y_prob, costs: CostAssumptions, steps: int = 199):
    """Expected cost across a grid of thresholds."""
    grid = np.linspace(0.01, 0.99, steps)
    return [CostCurvePoint(float(t), expected_cost(y_true, y_prob, t, costs)) for t in grid]


def best_threshold(y_true, y_prob, costs: CostAssumptions) -> float:
    """Threshold that minimises expected cost on the given data."""
    curve = cost_curve(y_true, y_prob, costs)
    return min(curve, key=lambda p: p.cost).threshold


def approve_all_cost(y_true, costs: CostAssumptions) -> float:
    """Baseline: no model, everyone is approved, so every default is a loss."""
    return float(np.sum(y_true) * costs.cost_false_negative)


def savings_summary(
    y_true, y_prob, threshold: float, costs: CostAssumptions
) -> dict[str, float]:
    """Compare the model at a threshold against approving everyone."""
    model_cost = expected_cost(y_true, y_prob, threshold, costs)
    baseline = approve_all_cost(y_true, costs)
    n = len(y_true)
    saved = baseline - model_cost
    return {
        "model_cost": float(model_cost),
        "approve_all_cost": float(baseline),
        "dollars_saved": float(saved),
        "saved_per_applicant": float(saved / n) if n else 0.0,
        "loss_reduction_pct": float(100.0 * saved / baseline) if baseline else 0.0,
    }
