"""Load the raw credit file and turn it into a modelling frame.

The public dataset ships with a handful of quirks that matter for a scorer:

* MonthlyIncome and NumberOfDependents have missing values.
* The three "past due" counters use 96 and 98 as data-entry sentinels for a
  small group of records. Left alone they look like someone was 96 months late.
* A single applicant has an age of 0.

I clean the obvious data-entry problems here but leave anything that could carry
signal (a missing income, say) to the model pipeline, where imputation is fit on
the training fold only. That keeps the split honest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET

RAW_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

SENTINEL_THRESHOLD = 90


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-entry errors that are clearly not real observations.

    Returns a new frame; the input is not mutated.
    """
    out = df.copy()

    # Sentinels of 96/98 in the delinquency counters are not real delays. Cap
    # them to the largest believable count observed below the sentinel band.
    for col in PAST_DUE_COLUMNS:
        believable = out.loc[out[col] < SENTINEL_THRESHOLD, col]
        ceiling = int(believable.max()) if not believable.empty else 0
        out[col] = np.where(out[col] >= SENTINEL_THRESHOLD, ceiling, out[col])

    # An age of 0 is impossible for a credit applicant. Push it to the adult
    # minimum so the row stays usable without inventing a precise age.
    out.loc[out["age"] < 18, "age"] = 18

    return out


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read the raw csv, dropping the unnamed row-id index column."""
    df = pd.read_csv(csv_path)
    id_like = [c for c in df.columns if c.lower() in {"id", "unnamed: 0"}]
    if id_like:
        df = df.drop(columns=id_like)
    return df


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return cleaned features X and target y from the raw file."""
    df = clean_frame(load_raw(csv_path))
    y = df[TARGET].astype(int)
    x = df.drop(columns=[TARGET])
    return x, y


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Stratified train / validation / test split.

    Stratifying on the target keeps the 6.7% default rate stable across all
    three folds, which matters when the positive class is this rare.
    """
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_fraction = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=val_fraction, stratify=y_dev, random_state=random_state
    )
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }
