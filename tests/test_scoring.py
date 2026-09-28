"""Scoring path and explanations, end to end through the public function."""

from __future__ import annotations

import math

from riskbeacon.explain import reasons_for
from riskbeacon.scoring import ApplicantRecord, risk_band, score_applicant


def test_risk_band_edges():
    assert risk_band(0.01) == "Low"
    assert risk_band(0.10) == "Moderate"
    assert risk_band(0.30) == "Elevated"
    assert risk_band(0.90) == "High"


def _good_applicant():
    return ApplicantRecord(
        RevolvingUtilizationOfUnsecuredLines=0.1,
        age=52,
        NumberOfTime30to59DaysPastDueNotWorse=0,
        DebtRatio=0.2,
        MonthlyIncome=8000.0,
        NumberOfOpenCreditLinesAndLoans=6,
        NumberOfTimes90DaysLate=0,
        NumberRealEstateLoansOrLines=1,
        NumberOfTime60to89DaysPastDueNotWorse=0,
        NumberOfDependents=1.0,
    )


def _risky_applicant():
    return ApplicantRecord(
        RevolvingUtilizationOfUnsecuredLines=1.4,
        age=23,
        NumberOfTime30to59DaysPastDueNotWorse=3,
        DebtRatio=0.9,
        MonthlyIncome=1500.0,
        NumberOfOpenCreditLinesAndLoans=12,
        NumberOfTimes90DaysLate=2,
        NumberRealEstateLoansOrLines=0,
        NumberOfTime60to89DaysPastDueNotWorse=1,
        NumberOfDependents=4.0,
    )


def test_score_applicant_returns_valid_probability(trained_bundle):
    result = score_applicant(_good_applicant(), trained_bundle)
    p = result["probability_of_default"]
    assert 0.0 <= p <= 1.0 and not math.isnan(p)
    assert result["decision"] in {"approve", "decline"}
    assert result["reasons"]


def test_risky_scores_above_safe(trained_bundle):
    safe = score_applicant(_good_applicant(), trained_bundle)
    risky = score_applicant(_risky_applicant(), trained_bundle)
    assert risky["probability_of_default"] > safe["probability_of_default"]


def test_missing_income_is_handled(trained_bundle):
    rec = _good_applicant()
    rec.MonthlyIncome = None
    result = score_applicant(rec, trained_bundle)
    assert 0.0 <= result["probability_of_default"] <= 1.0


def test_reasons_flag_severe_delinquency():
    fields = {"NumberOfTimes90DaysLate": 2, "RevolvingUtilizationOfUnsecuredLines": 1.3}
    reasons = reasons_for(fields)
    assert any("90" in r for r in reasons)


def test_reasons_clean_file_has_no_flags():
    fields = {
        "RevolvingUtilizationOfUnsecuredLines": 0.1,
        "age": 50,
        "MonthlyIncome": 9000,
        "DebtRatio": 0.2,
        "NumberOfTimes90DaysLate": 0,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
    }
    assert reasons_for(fields) == ["No standout risk factors in the application."]
