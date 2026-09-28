"""Gradio front end for scoring a single application.

The app is deliberately thin. Every number it shows comes from
`score_applicant`, the same function the tests cover, so the UI cannot drift away
from the scored logic.
"""

from __future__ import annotations

import gradio as gr

from .scoring import ApplicantRecord, load_bundle, score_applicant

_BUNDLE = None


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_bundle()
    return _BUNDLE


def evaluate(
    utilization,
    age,
    late_30_59,
    debt_ratio,
    monthly_income,
    open_lines,
    late_90,
    real_estate_lines,
    late_60_89,
    dependents,
):
    record = ApplicantRecord(
        RevolvingUtilizationOfUnsecuredLines=utilization,
        age=int(age),
        NumberOfTime30to59DaysPastDueNotWorse=int(late_30_59),
        DebtRatio=debt_ratio,
        MonthlyIncome=monthly_income if monthly_income and monthly_income > 0 else None,
        NumberOfOpenCreditLinesAndLoans=int(open_lines),
        NumberOfTimes90DaysLate=int(late_90),
        NumberRealEstateLoansOrLines=int(real_estate_lines),
        NumberOfTime60to89DaysPastDueNotWorse=int(late_60_89),
        NumberOfDependents=dependents,
    )
    result = score_applicant(record, _bundle())
    pct = result["probability_of_default"] * 100
    headline = (
        f"Default probability: {pct:.1f}%  |  Band: {result['risk_band']}  |  "
        f"Recommendation: {result['decision'].upper()}"
    )
    reasons = "\n".join(f"- {r}" for r in result["reasons"])
    return headline, reasons


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="riskbeacon") as demo:
        gr.Markdown("# riskbeacon\nEarly-warning default scoring for a consumer loan book.")
        with gr.Row():
            with gr.Column():
                utilization = gr.Number(label="Revolving utilisation (0-2)", value=0.35)
                age = gr.Number(label="Age", value=45)
                debt_ratio = gr.Number(label="Debt ratio", value=0.3)
                monthly_income = gr.Number(label="Monthly income (blank if unknown)", value=5400)
                dependents = gr.Number(label="Number of dependents", value=1)
            with gr.Column():
                open_lines = gr.Number(label="Open credit lines and loans", value=8)
                real_estate_lines = gr.Number(label="Real estate loans or lines", value=1)
                late_30_59 = gr.Number(label="Times 30-59 days past due", value=0)
                late_60_89 = gr.Number(label="Times 60-89 days past due", value=0)
                late_90 = gr.Number(label="Times 90+ days late", value=0)
        score_btn = gr.Button("Score application", variant="primary")
        headline = gr.Textbox(label="Result", interactive=False)
        reasons = gr.Textbox(label="Why", interactive=False, lines=6)
        score_btn.click(
            evaluate,
            inputs=[
                utilization, age, late_30_59, debt_ratio, monthly_income,
                open_lines, late_90, real_estate_lines, late_60_89, dependents,
            ],
            outputs=[headline, reasons],
        )
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
