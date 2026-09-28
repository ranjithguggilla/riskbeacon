"""Chart generation for the README and the dossier.

Matplotlib only, saved as png into assets/. The functions take arrays already
computed by the evaluation code so plotting stays a dumb rendering step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    precision_recall_curve,
    roc_curve,
)

from .config import CostAssumptions  # noqa: E402
from .evaluate import cost_curve  # noqa: E402

NAVY = "#1f3a5f"
AMBER = "#e08a1e"
SLATE = "#6b7280"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=NAVY)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_roc(y_true, y_prob, out: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(fpr, tpr, color=NAVY, lw=2)
    ax.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1)
    _style(ax, "ROC curve on held-out test set", "False positive rate", "True positive rate")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_pr(y_true, y_prob, out: Path) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    base = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(recall, precision, color=AMBER, lw=2)
    ax.axhline(base, color=SLATE, ls="--", lw=1, label=f"base rate {base:.3f}")
    _style(ax, "Precision-recall on held-out test set", "Recall", "Precision")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_cost_curve(y_true, y_prob, costs: CostAssumptions, chosen: float, out: Path) -> Path:
    curve = cost_curve(y_true, y_prob, costs)
    xs = [p.threshold for p in curve]
    ys = [p.cost for p in curve]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(xs, ys, color=NAVY, lw=2)
    ax.axvline(chosen, color=AMBER, ls="--", lw=1.5, label=f"chosen {chosen:.2f}")
    _style(ax, "Expected cost by decision threshold", "Threshold", "Expected cost (USD)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_feature_importance(names, importances, out: Path, top: int = 12) -> Path:
    order = np.argsort(importances)[::-1][:top]
    names = np.asarray(names)[order][::-1]
    vals = np.asarray(importances)[order][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(names, vals, color=NAVY)
    _style(ax, "Feature importance (winning model)", "Importance", "")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_model_comparison(names, aucs, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = [AMBER if a == max(aucs) else NAVY for a in aucs]
    ax.bar(names, aucs, color=colors)
    ax.set_ylim(0.5, max(aucs) + 0.03)
    _style(ax, "Validation ROC AUC by model", "", "ROC AUC")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
