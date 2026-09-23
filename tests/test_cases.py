"""Date facts preserve evidence; a numeric sentence does not justify a uniform label."""

from syn.cases import date_facts, extra_cases
from syn.schema import EvalExample, Option, ScoreRequest


def _row(context: str) -> EvalExample:
    return EvalExample(
        request=ScoreRequest(
            context=context,
            question="Which tier applies?",
            options=[
                Option(id="list", text="List price"),
                Option(id="volume", text="Volume discount"),
            ],
        ),
        expected_option_id="list",
        source="pricing",
    )


def test_date_facts_compare_each_later_date_with_each_earlier_one():
    text = "Due July 4, 2026. Received June 26, 2026. Shipped 2026-07-01."
    assert date_facts(text) == (
        "June 26, 2026 is 8 days before July 4, 2026. "
        "2026-07-01 is 3 days before July 4, 2026. "
        "2026-07-01 is 5 days after June 26, 2026."
    )
    assert date_facts("Only July 4, 2026.") == ""


def test_extra_cases_never_invent_uncertainty_from_a_numeric_sentence():
    row = _row(
        "policy: A reply within 4 hours meets the service level. "
        "case: The ticket was opened. The first response arrived 17 hours later."
    )
    assert extra_cases([row]) == []
    irrelevant = _row(
        "policy: Orders below 25 units pay list price. Otherwise use the volume discount. "
        "The routing reference does not affect pricing. "
        "case: The routing reference is 901. The order is for 10 units."
    )
    assert extra_cases([irrelevant]) == []


def test_date_copy_preserves_label_and_all_evidence():
    dated_row = _row("Due July 4, 2026. Received June 26, 2026. case: no number here.")
    copies = extra_cases([dated_row])
    assert len(copies) == 1
    assert "date_facts:" in copies[0].request.context
    assert "8 days before" in copies[0].request.context
    assert copies[0].uniform is False
    assert copies[0].request.context.startswith(dated_row.request.context + "\n\n")
    assert copies[0].expected_option_id == dated_row.expected_option_id
    assert copies[0].source == dated_row.source
    assert extra_cases(copies) == []
