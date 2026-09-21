import pytest

from syn.bench import _answer, target_metrics, to_systemone
from syn.schema import ScoreRequest

REQUEST = ScoreRequest.model_validate(
    {
        "context": "none of it amounts to much",
        "question": "Sentiment?",
        "options": [{"id": str(i), "text": t} for i, t in enumerate(["bad", "meh", "good"])],
    }
)


def test_an_ordinal_row_becomes_a_score_question():
    body = to_systemone(REQUEST, "m", ordinal=True)
    assert body["questions"]["answer"] == {
        "type": "score",
        "instructions": "Sentiment?",
        "criteria": ["bad", "meh", "good"],
    }
    assert to_systemone(REQUEST, "m")["questions"]["answer"]["type"] == "choice"


def test_score_answers_map_levels_back_to_options():
    body = {
        "model": "m",
        "answers": {
            "answer": {
                "type": "score",
                "score": 0.9,
                "confidence": 0.5,
                "legend": {"0": "bad", "1": "meh", "2": "good"},
                "probabilities": {"0": 0.3, "1": 0.6, "2": 0.1},
            }
        },
        "usage": {"input_tokens": 12},
    }
    answer = _answer(body, REQUEST, ordinal=True)
    assert answer["choice"] == "1" and answer["score"] == 0.9
    assert answer["probabilities"] == {"0": 0.3, "1": 0.6, "2": 0.1}
    body["answers"]["answer"]["probabilities"] = {"0": 0.5, "1": 0.5}
    with pytest.raises(ValueError, match="levels"):
        _answer(body, REQUEST, ordinal=True)


def test_target_metrics_report_level_error_for_ordinal_rows():
    rows = [
        {
            "choice": "1",
            "expected_option_id": "1",
            "probabilities": {"0": 0.2, "1": 0.7, "2": 0.1},
            "latency_ms": 5.0,
            "score": 0.9,
            "expected_level": 1,
        },
        {
            "choice": "2",
            "expected_option_id": "0",
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            "latency_ms": 5.0,
            "score": 1.6,
            "expected_level": 0,
        },
        {
            "choice": "a",
            "expected_option_id": "a",
            "probabilities": {"a": 0.9, "b": 0.1},
            "latency_ms": 5.0,
        },
    ]
    metrics = target_metrics(rows, bootstrap=0)
    assert metrics["ordinal_rows"] == 2
    assert metrics["level_mae"] == pytest.approx((0.1 + 1.6) / 2)
    assert target_metrics(rows[2:], bootstrap=0)["level_mae"] is None
