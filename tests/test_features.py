"""Feature engineering behaviour and leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskbeacon.features import FeatureEngineer, engineer


def _sample():
    return pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.2, 1.5],
            "age": [30, 55],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 2],
            "DebtRatio": [0.3, 8.0],
            "MonthlyIncome": [5000.0, np.nan],
            "NumberOfOpenCreditLinesAndLoans": [5, 9],
            "NumberOfTimes90DaysLate": [0, 1],
            "NumberRealEstateLoansOrLines": [1, 2],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0],
            "NumberOfDependents": [1.0, 3.0],
        }
    )


def test_missing_income_flag_set():
    out = engineer(_sample())
    assert out["MissingIncome"].tolist() == [0, 1]


def test_derived_columns_present():
    out = engineer(_sample())
    for col in [
        "IncomePerDependent",
        "TotalTimesPastDue",
        "HasSevereDelinquency",
        "UtilizationCapped",
        "HighUtilization",
        "DebtRatioCapped",
    ]:
        assert col in out.columns


def test_utilization_and_debt_ratio_capped():
    out = engineer(_sample())
    assert out["UtilizationCapped"].max() <= 2.0
    assert out["DebtRatioCapped"].max() <= 5.0
    assert out["HighUtilization"].tolist() == [0, 1]


def test_total_past_due_adds_up():
    out = engineer(_sample())
    assert out["TotalTimesPastDue"].tolist() == [0, 3]


def test_transform_is_row_independent():
    df = _sample()
    single = engineer(df.iloc[[1]]).reset_index(drop=True)
    batch = engineer(df).iloc[[1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(single, batch)


def test_engineer_does_not_mutate_input():
    df = _sample()
    before = df.copy()
    FeatureEngineer().fit_transform(df)
    pd.testing.assert_frame_equal(df, before)
