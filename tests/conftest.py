"""Shared fixtures. Trains a small, fast pipeline so the scoring tests do not
depend on a full training run or a committed model artifact.
"""

from __future__ import annotations

import pytest

from riskbeacon.config import Paths
from riskbeacon.data import load_dataset, split_dataset
from riskbeacon.fetch_data import main as fetch_data
from riskbeacon.model import build_logistic_regression


@pytest.fixture(scope="session")
def raw_split():
    if not Paths().raw_csv.exists():
        fetch_data()
    x, y = load_dataset(Paths().raw_csv)
    # A small stratified slice keeps the suite quick while staying realistic.
    x_small = x.sample(4000, random_state=0)
    y_small = y.loc[x_small.index]
    return split_dataset(x_small, y_small)


@pytest.fixture(scope="session")
def trained_bundle(raw_split):
    x_train, y_train = raw_split["train"]
    pipe = build_logistic_regression()
    pipe.fit(x_train, y_train)
    return {"pipeline": pipe, "threshold": 0.5, "winner": "logistic_regression"}
