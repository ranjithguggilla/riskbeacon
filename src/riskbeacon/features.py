"""Feature engineering that runs inside the model pipeline.

Everything here is row-wise and stateless, so it holds no statistics learned
from the data. That is on purpose: it means the transformer cannot leak
information between the train and test folds. Anything that does need to learn
from data (median imputation) lives after this step and is fit on the training
fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

UTILIZATION_CAP = 2.0
DEBT_RATIO_CAP = 5.0


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that a credit analyst would reach for.

    The new columns capture three ideas the raw fields only hint at: whether an
    income figure was reported at all, how deep the delinquency history runs,
    and how stretched the borrower looks once income is spread across
    dependents.
    """
    out = df.copy()
    out["MonthlyIncome"] = pd.to_numeric(out["MonthlyIncome"], errors="coerce")
    out["NumberOfDependents"] = pd.to_numeric(out["NumberOfDependents"], errors="coerce")

    out["MissingIncome"] = out["MonthlyIncome"].isna().astype(int)
    out["MissingDependents"] = out["NumberOfDependents"].isna().astype(int)

    dependents = out["NumberOfDependents"].fillna(0)
    out["IncomePerDependent"] = out["MonthlyIncome"] / (dependents + 1.0)

    out["TotalTimesPastDue"] = out[PAST_DUE_COLUMNS].sum(axis=1)
    out["HasSevereDelinquency"] = (out["NumberOfTimes90DaysLate"] > 0).astype(int)

    out["UtilizationCapped"] = out["RevolvingUtilizationOfUnsecuredLines"].clip(
        upper=UTILIZATION_CAP
    )
    out["HighUtilization"] = (
        out["RevolvingUtilizationOfUnsecuredLines"] > 1.0
    ).astype(int)
    out["DebtRatioCapped"] = out["DebtRatio"].clip(upper=DEBT_RATIO_CAP)

    out["MonthlyDebtPayment"] = out["DebtRatioCapped"] * out["MonthlyIncome"].fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["RealEstateShare"] = out["NumberRealEstateLoansOrLines"] / (
            out["NumberOfOpenCreditLinesAndLoans"] + 1.0
        )

    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn wrapper around :func:`engineer` so it slots into a Pipeline."""

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D102, ARG002
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        return engineer(x)

    def get_feature_names_out(self, input_features=None):  # noqa: D102, ARG002
        return None
