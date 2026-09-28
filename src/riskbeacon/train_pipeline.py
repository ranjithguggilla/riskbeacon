"""End-to-end training run: load, compare, pick, evaluate, persist.

Run it with `python -m riskbeacon.train_pipeline`. It trains every candidate,
selects the winner on validation ROC AUC, sets the decision threshold by minimum
expected cost on the validation fold, then reports honest numbers on a test set
the model never saw during selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .config import CostAssumptions, Paths
from .data import load_dataset, split_dataset
from .evaluate import (
    best_threshold,
    ranking_metrics,
    savings_summary,
    threshold_metrics,
)
from .model import candidate_models
from .plots import (
    plot_cost_curve,
    plot_feature_importance,
    plot_model_comparison,
    plot_pr,
    plot_roc,
)


def _feature_importance(pipeline, feature_names) -> dict[str, float] | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "get_feature_importance"):
        vals = model.get_feature_importance()
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return None
    return {name: float(v) for name, v in zip(feature_names, vals, strict=False)}


def run(paths: Paths | None = None, costs: CostAssumptions | None = None) -> dict:
    paths = paths or Paths()
    costs = costs or CostAssumptions()
    paths.models.mkdir(parents=True, exist_ok=True)
    paths.assets.mkdir(parents=True, exist_ok=True)

    x, y = load_dataset(paths.raw_csv)
    folds = split_dataset(x, y)
    x_train, y_train = folds["train"]
    x_val, y_val = folds["val"]
    x_test, y_test = folds["test"]

    results = {}
    fitted = {}
    for name, pipe in candidate_models().items():
        pipe.fit(x_train, y_train)
        val_prob = pipe.predict_proba(x_val)[:, 1]
        results[name] = ranking_metrics(y_val, val_prob)
        fitted[name] = pipe
        print(f"{name:22s} val ROC AUC {results[name]['roc_auc']:.4f}")

    winner_name = max(results, key=lambda n: results[n]["roc_auc"])
    winner = fitted[winner_name]
    print(f"winner: {winner_name}")

    val_prob = winner.predict_proba(x_val)[:, 1]
    threshold = best_threshold(y_val, val_prob, costs)

    test_prob = winner.predict_proba(x_test)[:, 1]
    test_ranking = ranking_metrics(y_test, test_prob)
    test_threshold = threshold_metrics(y_test, test_prob, threshold)
    test_savings = savings_summary(y_test, test_prob, threshold, costs)

    # Feature names come out of the engineer applied to the training frame.
    engineered = winner.named_steps["features"].transform(x_train)
    feature_names = list(engineered.columns)
    importance = _feature_importance(winner, feature_names)

    # Charts.
    plot_roc(y_test, test_prob, paths.assets / "roc_curve.png")
    plot_pr(y_test, test_prob, paths.assets / "pr_curve.png")
    plot_cost_curve(y_test, test_prob, costs, threshold, paths.assets / "cost_curve.png")
    plot_model_comparison(
        list(results.keys()),
        [results[n]["roc_auc"] for n in results],
        paths.assets / "model_comparison.png",
    )
    if importance:
        plot_feature_importance(
            list(importance.keys()),
            list(importance.values()),
            paths.assets / "feature_importance.png",
        )

    bundle = {
        "pipeline": winner,
        "winner": winner_name,
        "threshold": threshold,
        "feature_importance": importance,
        "costs": costs.__dict__,
    }
    joblib.dump(bundle, paths.model_file)

    metrics = {
        "winner": winner_name,
        "validation_roc_auc": {n: results[n]["roc_auc"] for n in results},
        "validation_average_precision": {
            n: results[n]["average_precision"] for n in results
        },
        "chosen_threshold": threshold,
        "test": {
            "ranking": test_ranking,
            "at_threshold": test_threshold,
            "business": test_savings,
        },
        "cost_assumptions": {
            "exposure_at_default": costs.exposure_at_default,
            "loss_given_default": costs.loss_given_default,
            "approved_margin": costs.approved_margin,
            "cost_false_negative": costs.cost_false_negative,
            "cost_false_positive": costs.cost_false_positive,
        },
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "default_rate": float(y.mean()),
    }
    Path(paths.metrics).write_text(json.dumps(metrics, indent=2))
    print(f"test ROC AUC {test_ranking['roc_auc']:.4f}")
    print(f"dollars saved on test fold {test_savings['dollars_saved']:,.0f}")
    return metrics


if __name__ == "__main__":
    run()
