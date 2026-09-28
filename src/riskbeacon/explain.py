"""Plain-language reasons behind a risk score.

A probability on its own is not much use to a loan officer who has to justify a
decline. These rules turn the raw application fields into short, human sentences
that point at what drove the score up. They are intentionally simple and
transparent rather than a post-hoc attribution method, so anyone can audit why a
reason fired.
"""

from __future__ import annotations

from collections.abc import Mapping


def reasons_for(fields: Mapping[str, float]) -> list[str]:
    """Return the notable risk factors in an application, worst first."""
    notes: list[tuple[int, str]] = []

    util = fields.get("RevolvingUtilizationOfUnsecuredLines", 0.0) or 0.0
    if util > 1.0:
        notes.append((5, "Revolving balances exceed the available credit limit."))
    elif util > 0.6:
        notes.append((3, "High revolving utilisation, above 60% of the limit."))

    late_90 = fields.get("NumberOfTimes90DaysLate", 0) or 0
    if late_90 >= 1:
        notes.append((5, f"Has been 90+ days late {int(late_90)} time(s)."))

    late_30 = fields.get("NumberOfTime30-59DaysPastDueNotWorse", 0) or 0
    late_60 = fields.get("NumberOfTime60-89DaysPastDueNotWorse", 0) or 0
    if late_30 + late_60 >= 2:
        notes.append((3, "Repeated 30 to 89 day payment delays on record."))

    debt_ratio = fields.get("DebtRatio", 0.0) or 0.0
    income = fields.get("MonthlyIncome")
    if income is not None and income > 0 and debt_ratio > 0.5:
        notes.append((3, "Monthly debt payments are more than half of income."))

    if income is None or (isinstance(income, float) and income != income):
        notes.append((2, "No monthly income was reported."))
    elif income is not None and income < 2000:
        notes.append((2, "Reported monthly income is low."))

    age = fields.get("age", 0) or 0
    if 18 <= age < 25:
        notes.append((1, "Thin file expected for a younger applicant."))

    dependents = fields.get("NumberOfDependents", 0) or 0
    if dependents >= 4:
        notes.append((1, "Large number of dependents adds to fixed outgoings."))

    if not notes:
        notes.append((0, "No standout risk factors in the application."))

    notes.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in notes]
