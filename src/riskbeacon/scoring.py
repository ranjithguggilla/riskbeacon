"""The single entry point the UI and tests both call to score an applicant.

Keeping this logic in one small, importable function means the Gradio app is a
thin shell over code that is exercised directly by the test suite. Nothing about
a decision lives only inside the app.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import Paths
from .data import RAW_COLUMNS, clean_frame
from .explain import reasons_for

RISK_BANDS = [
    (0.05, "Low"),
    (0.15, "Moderate"),
    (0.35, "Elevated"),
    (1.01, "High"),
]


@dataclass
class ApplicantRecord:
    """One credit application, in the raw units the dataset uses."""

    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30to59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float | None
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60to89DaysPastDueNotWorse: int
    NumberOfDependents: float | None

    def to_row(self) -> dict[str, float]:
        """Map the record onto the exact column names the model expects."""
        data = asdict(self)
        return {
            "RevolvingUtilizationOfUnsecuredLines": data[
                "RevolvingUtilizationOfUnsecuredLines"
            ],
            "age": data["age"],
            "NumberOfTime30-59DaysPastDueNotWorse": data[
                "NumberOfTime30to59DaysPastDueNotWorse"
            ],
            "DebtRatio": data["DebtRatio"],
            "MonthlyIncome": data["MonthlyIncome"],
            "NumberOfOpenCreditLinesAndLoans": data["NumberOfOpenCreditLinesAndLoans"],
            "NumberOfTimes90DaysLate": data["NumberOfTimes90DaysLate"],
            "NumberRealEstateLoansOrLines": data["NumberRealEstateLoansOrLines"],
            "NumberOfTime60-89DaysPastDueNotWorse": data[
                "NumberOfTime60to89DaysPastDueNotWorse"
            ],
            "NumberOfDependents": data["NumberOfDependents"],
        }


def risk_band(probability: float) -> str:
    """Bucket a probability into a named band for quick reading."""
    for edge, label in RISK_BANDS:
        if probability < edge:
            return label
    return "High"


def load_bundle(model_file: Path | None = None) -> dict:
    """Load the trained pipeline and its decision threshold from disk."""
    path = model_file or Paths().model_file
    return joblib.load(path)


def score_applicant(record: ApplicantRecord, bundle: Mapping) -> dict:
    """Score one applicant and explain the outcome.

    Returns the default probability, the approve / decline call at the model's
    chosen threshold, a readable risk band, and the reasons behind the score.
    """
    row = record.to_row()
    frame = clean_frame(pd.DataFrame([row]))[RAW_COLUMNS]
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])

    probability = float(pipeline.predict_proba(frame)[0, 1])
    decline = probability >= threshold

    return {
        "probability_of_default": probability,
        "decision": "decline" if decline else "approve",
        "risk_band": risk_band(probability),
        "threshold": threshold,
        "reasons": reasons_for(row),
    }

