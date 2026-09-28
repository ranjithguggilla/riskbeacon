"""riskbeacon: an early-warning credit default scorer for consumer lending."""

from .config import CostAssumptions, Paths
from .scoring import ApplicantRecord, score_applicant

__all__ = [
    "ApplicantRecord",
    "CostAssumptions",
    "Paths",
    "score_applicant",
]

__version__ = "0.1.0"
