"""Project paths and the money assumptions behind the cost analysis.

The cost numbers here are deliberately explicit and easy to change. They are the
levers a lending team would argue about in a review, so I keep them in one place
rather than scattering magic numbers through the scoring code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_CSV_NAME = "cs-training.csv"
TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42


@dataclass(frozen=True)
class Paths:
    """Filesystem layout for inputs and generated artifacts."""

    root: Path = ROOT
    data: Path = ROOT / "data"
    models: Path = ROOT / "models"
    assets: Path = ROOT / "assets"
    metrics: Path = ROOT / "metrics.json"

    @property
    def raw_csv(self) -> Path:
        return self.data / RAW_CSV_NAME

    @property
    def model_file(self) -> Path:
        return self.models / "riskbeacon.joblib"


@dataclass(frozen=True)
class CostAssumptions:
    """What a wrong decision costs, in dollars.

    exposure_at_default: average unsecured balance carried by a borrower.
    loss_given_default: fraction of that balance we do not recover in a default.
    approved_margin: two-year net margin earned on a borrower who stays current.

    A missed default (false negative) costs exposure * loss_given_default.
    A declined good customer (false positive) costs the margin we walk away from.
    """

    exposure_at_default: float = 10_000.0
    loss_given_default: float = 0.70
    approved_margin: float = 600.0

    @property
    def cost_false_negative(self) -> float:
        return self.exposure_at_default * self.loss_given_default

    @property
    def cost_false_positive(self) -> float:
        return self.approved_margin
