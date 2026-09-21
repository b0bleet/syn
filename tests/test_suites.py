import json

import pytest

from syn import cli
from syn.evaluation import read_jsonl
from syn.schema import EvalExample
from syn.suites import convert, convert_record

NOUL = {
    "state": "The passage says the team won in 2018.",
    "questions": {
        "answer": {
            "type": "noul",
            "instructions": "Did the team win?",
            "label": True,
            "src": "facts",
            "criteria": {"true": "The passage supports yes", "false": "It does not"},
        }
    },
    "_meta": {"source": "wiki"},
}
SCORE = {
    "state": {"review": "none of it amounts to much"},
    "questions": {
        "sentiment": {
            "type": "score",
            "instructions": {"question": "Sentiment?", "focus": "Pick one."},
            "criteria": ["very negative", "negative", "neutral", "positive"],
            "label": 1,
        },
        "topic": {
            "type": "choice",
            "instructions": "Topic?",
            "criteria": {"film": None, "food": "Restaurants"},
            "label": "film",
        },
        "unlabelled": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}},
        "soft": {
            "type": "noul",
            "instructions": "Unknowable?",
            "label": "maybe",
            "target": {"true": 0.5, "false": 0.5},
        },
        "lonely": {
            "type": "choice",
            "instructions": "?",
            "criteria": {"only": None},
            "label": "only",
        },
        "wrong_key": {
            "type": "choice",
            "instructions": "?",
            "criteria": {"a": None, "b": None},
            "label": "c",
        },
        "bad_type": {"type": "rank", "instructions": "?", "criteria": {"a": None}, "label": "a"},
    },
    "_meta": {"source": "reviews"},
}


def test_convert_record_renders_like_the_service_and_tags_rows():
    rows, skipped = convert_record(NOUL, "fallback")
    assert len(rows) == 1 and not skipped
    row = EvalExample.model_validate(rows[0])
    assert row.expected_option_id == "yes" and row.source == "facts" and not row.ordinal
    assert [o.text for o in row.request.options] == [
        "Yes: The passage supports yes",
        "No: It does not",
    ]
    assert row.request.question == "Did the team win?"

    rows, skipped = convert_record(SCORE, "fallback")
    by_question = {r["request"]["question"][:9]: r for r in rows}
    assert len(rows) == 2
    score = EvalExample.model_validate(by_question["question:"])
    assert score.ordinal and score.expected_option_id == "1" and score.source == "reviews"
    assert score.request.context == "review: none of it amounts to much"
    assert score.request.question == "question: Sentiment?\nfocus: Pick one."
    assert [o.id for o in score.request.options] == ["0", "1", "2", "3"]
    topic = EvalExample.model_validate(by_question["Topic?"])
    assert topic.expected_option_id == "film" and not topic.ordinal
    assert [o.text for o in topic.request.options] == ["film", "food: Restaurants"]
    assert skipped == {
        "unlabelled": 1,
        "label_names_no_option": 2,
        "single_option": 1,
        "invalid_question": 1,
    }
    assert convert_record("not a record", "x") == ([], {"malformed_record": 1})
    assert convert_record({"state": "s", "questions": {"q": 1}}, "x")[1] == {
        "malformed_question": 1
    }


def test_convert_suite_directory_renames_splits_and_refuses_overwrites(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "train.jsonl").write_text(json.dumps(NOUL) + "\n" + json.dumps(SCORE) + "\n")
    (suite / "development.jsonl").write_text(json.dumps(NOUL) + "\n")
    out = tmp_path / "data" / "suite"
    report = convert([suite], out)
    assert set(report["files"]) == {str(suite / "train.jsonl"), str(suite / "development.jsonl")}
    assert (out / "train.jsonl").exists() and (out / "validation.jsonl").exists()
    train = report["files"][str(suite / "train.jsonl")]
    assert train["rows"] == 3 and train["ordinal_rows"] == 1
    assert train["sources"] == ["facts", "reviews"]
    assert train["skipped"]["unlabelled"] == 1
    assert [EvalExample.model_validate(r) for r in read_jsonl(out / "validation.jsonl")]
    with pytest.raises(FileExistsError):
        convert([suite / "development.jsonl"], out)
    # A record without any source falls back to the directory name, or --source.
    bare = tmp_path / "bare" / "rows.jsonl"
    bare.parent.mkdir()
    bare.write_text(
        json.dumps({"state": "s", "questions": {"q": NOUL["questions"]["answer"] | {"src": None}}})
        + "\n"
    )
    assert convert([bare], tmp_path / "o1")["files"][str(bare)]["sources"] == ["bare"]
    assert convert([bare], tmp_path / "o2", source="mine")["files"][str(bare)]["sources"] == [
        "mine"
    ]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No .jsonl"):
        convert([tmp_path / "empty"], tmp_path / "o3")
    with pytest.raises(FileNotFoundError):
        convert([tmp_path / "nothing"], tmp_path / "o4")


def test_cli_import_systemone(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        "syn.suites.convert",
        lambda paths, out, source: (
            seen.setdefault("call", ([str(p) for p in paths], str(out), source)),
            {"ok": 1},
        )[1],
    )
    cli.main(["import-systemone", "a", "b.jsonl", "--out", "data/x", "--source", "s"])
    assert seen["call"] == (["a", "b.jsonl"], "data/x", "s")
    assert json.loads(capsys.readouterr().out) == {"ok": 1}
