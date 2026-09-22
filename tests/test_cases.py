"""Day-count sentences and withheld policy cases."""

from syn.cases import date_facts, extra_cases, withhold_case
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


def test_extra_cases_add_a_dated_copy_and_a_uniform_copy():
    row = _row(
        "policy: A reply within 4 hours meets the service level. "
        "case: The ticket was opened. The first response arrived 17 hours later."
    )
    extra = extra_cases([row])
    dated = [item for item in extra if not item.uniform]
    withheld = [item for item in extra if item.uniform]
    assert dated == []
    assert len(withheld) == 1
    assert "17 hours" not in withheld[0].request.context
    assert "within 4 hours" in withheld[0].request.context
    assert withheld[0].expected_option_id == "list"

    dated_row = _row("Due July 4, 2026. Received June 26, 2026. case: no number here.")
    # the case sentence has no digit, so only the date-count copy is added
    assert withhold_case(dated_row.request.context) is None
    copies = extra_cases([dated_row])
    assert len(copies) == 1
    assert "date_facts:" in copies[0].request.context
    assert "8 days before" in copies[0].request.context
    assert copies[0].uniform is False
