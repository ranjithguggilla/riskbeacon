"""Candidate models, each wrapped so preprocessing cannot leak.

Three pipelines compete on the same folds:

* logistic_regression: a scaled, class-weighted linear baseline. It is the model
  I would ship if nothing else clearly beat it, because it is transparent and
  cheap to reason about.
* catboost_weighted: gradient boosting with balanced class weights.
* catboost_smote: the same booster, but the training fold is rebalanced with
  SMOTE first.

Every pipeline starts with the feature engineer and a median imputer so that all
preprocessing is fit inside cross-validation, never on the held-out data.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .features import FeatureEngineer


def _catboost(**overrides) -> CatBoostClassifier:
    params = {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(overrides)
    return CatBoostClassifier(**params)


def build_logistic_regression() -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_catboost_weighted() -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("impute", SimpleImputer(strategy="median")),
            ("model", _catboost(auto_class_weights="Balanced")),
        ]
    )


def build_catboost_smote() -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("impute", SimpleImputer(strategy="median")),
            ("resample", SMOTE(random_state=RANDOM_STATE)),
            ("model", _catboost()),
        ]
    )


def candidate_models() -> dict[str, ImbPipeline]:
    """Named registry of every pipeline the training run compares."""
    return {
        "logistic_regression": build_logistic_regression(),
        "catboost_weighted": build_catboost_weighted(),
        "catboost_smote": build_catboost_smote(),
    }
