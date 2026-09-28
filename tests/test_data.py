"""Data loading and cleaning behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskbeacon.config import TARGET, Paths
from riskbeacon.data import (
    PAST_DUE_COLUMNS,
    clean_frame,
    load_dataset,
    split_dataset,
)


def test_sentinels_are_capped():
    df = pd.DataFrame(
        {
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 98, 96],
            "NumberOfTimes90DaysLate": [0, 2, 98, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 96, 3],
            "age": [40, 40, 40, 40],
        }
    )
    cleaned = clean_frame(df)
    for col in PAST_DUE_COLUMNS:
        assert cleaned[col].max() < 90


def test_impossible_age_is_lifted_to_adult():
    df = pd.DataFrame(
        {c: [0, 0] for c in PAST_DUE_COLUMNS} | {"age": [0, 21]}
    )
    cleaned = clean_frame(df)
    assert cleaned["age"].min() == 18


def test_clean_frame_does_not_mutate_input():
    df = pd.DataFrame({c: [98] for c in PAST_DUE_COLUMNS} | {"age": [0]})
    before = df.copy()
    clean_frame(df)
    pd.testing.assert_frame_equal(df, before)


def test_load_dataset_separates_target():
    x, y = load_dataset(Paths().raw_csv)
    assert TARGET not in x.columns
    assert set(np.unique(y)) <= {0, 1}
    assert len(x) == len(y)


def test_split_preserves_default_rate():
    x, y = load_dataset(Paths().raw_csv)
    folds = split_dataset(x, y)
    rates = [folds[k][1].mean() for k in ("train", "val", "test")]
    assert max(rates) - min(rates) < 0.01
    total = sum(len(folds[k][0]) for k in folds)
    assert total == len(x)
