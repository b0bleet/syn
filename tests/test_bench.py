import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient
from helpers import CharacterTokenizer, ContentBackend, make_scorer

from syn import cli
from syn.api import create_app
from syn.bench import TargetError, bench, parse_target, run_target, to_systemone
from syn.config import Settings
from syn.schema import EvalExample, ScoreRequest
from syn.systemone import SystemOneRequest

BASE = {
    "question": "Choose a team",
    "options": [{"id": "sales", "text": "Sales"}, {"id": "billing", "text": "Billing"}],
}
ROWS = [
    {"request": {"context": "Duplicate charge", **BASE}, "expected_option_id": "billing"},
    {"request": {"context": "Wants a quote", **BASE}, "expected_option_id": "sales"},
    # Past the local service's token limit, so it answers 422 while the fake Jev still scores it.
    {"request": {"context": "x" * 9000, **BASE}, "expected_option_id": "billing"},
]


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def jev_handler(calls):
    """Fake Jev: right on both short rows, wrong on the long one."""

    def handle(request):
        calls.append(request)
        if request.headers.get("authorization") != "Bearer jev-key":
            return httpx.Response(403, json={"detail": "Must supply an API key!"})
        body = json.loads(request.content)
        ids = list(body["questions"]["answer"]["criteria"])
        pick = "billing" if "charge" in body["state"] else "sales"
        probabilities = {i: 0.9 if i == pick else 0.1 for i in ids}
        answer = {
            "type": "choice",
            "choice": pick,
            "confidence": 0.8,
            "probabilities": probabilities,
        }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"answer": answer},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

    return handle


def local_app():
    """This service, always picking sales, behind a bearer key."""
    tokenizer = CharacterTokenizer()
    backend = ContentBackend(tokenizer, {"Sales": math.log(0.6), "Billing": math.log(0.2)})
    settings = Settings(api_key="local-key")
    return create_app(settings, make_scorer(settings, backend, tokenizer))


def test_parse_target_presets_and_urls():
    jev = parse_target("jev")
    assert (jev.url, jev.model, jev.key_env, jev.key_required) == (
        "https://api.typesafe.ai",
        "jev-latest",
        "TYPESAFE_API_KEY",
        True,
    )
    assert parse_target("jev@jev-1.13.0").model == "jev-1.13.0"
    syn = parse_target("syn=http://127.0.0.1:8765/")
    assert (syn.url, syn.model, syn.key_env, syn.key_required) == (
        "http://127.0.0.1:8765",
        "syn-latest",
        "SYN_API_KEY",
        False,
    )
    proxy = parse_target("my-proxy@m1=https://proxy.test/x?a=b")
    assert (proxy.url, proxy.model, proxy.key_env) == (
        "https://proxy.test/x?a=b",
        "m1",
        "MY_PROXY_API_KEY",
    )
    for bad in ("acme", "Jev", "=http://x", "a b=http://x"):
        with pytest.raises(ValueError):
            parse_target(bad)


def test_to_systemone_is_one_choice_question_this_service_accepts():
    request = ScoreRequest.model_validate(
        {
            "context": "Claim your prize",
            "question": "Which label fits?",
            "criteria": "Promotions count as spam.",
            "options": [{"id": "spam", "text": "spam"}, {"id": "ham", "text": "Personal mail"}],
        }
    )
    body = to_systemone(request, "jev-latest")
    assert body == {
        "state": "Claim your prize",
        "model": "jev-latest",
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Which label fits?\n\nPromotions count as spam.",
                "criteria": {"spam": None, "ham": "Personal mail"},
            }
        },
    }
    SystemOneRequest.model_validate(body)


def test_bench_pairs_jev_with_this_service_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-key")
    monkeypatch.setenv("SYN_API_KEY", "local-key")
    dataset = tmp_path / "eval" / "routing.jsonl"
    dataset.parent.mkdir()
    write(dataset, ROWS)
    targets = [parse_target("jev"), parse_target("syn=http://testserver")]
    calls = []
    jev = httpx.Client(transport=httpx.MockTransport(jev_handler(calls)), base_url="http://jev/")
    out = tmp_path / "bench"
    with TestClient(local_app()) as syn:
        summary = bench([dataset], out, targets, 2, clients={"jev": jev, "syn": syn})
        assert len(calls) == 4  # warmup + 3 rows

        result = summary["datasets"]["routing"]
        assert summary["reference"] == "jev" and result["examples"] == 3
        j, s = result["targets"]["jev"], result["targets"]["syn"]
        assert (j["scored"], j["errors"], j["top1_accuracy"]) == (3, 0, pytest.approx(2 / 3))
        assert j["models"] == ["jev-1.13.0"] and j["input_tokens"] == 30
        assert j["negative_log_likelihood"] == pytest.approx(
            -(2 * math.log(0.9) + math.log(0.1)) / 3
        )
        assert j["multiclass_brier"] == pytest.approx((0.02 + 0.02 + 1.62) / 3)
        assert (s["scored"], s["errors"], s["error_statuses"]) == (2, 1, {"422": 1})
        assert s["top1_accuracy"] == 0.5 and s["models"] == ["Qwen/Qwen3-0.6B"]
        pair = result["paired_vs_reference"]["syn"]
        assert (pair["examples"], pair["accuracy_reference"], pair["accuracy"]) == (2, 1.0, 0.5)
        assert pair["difference_b_minus_a"] == -0.5 and pair["only_a_correct"] == 1
        assert pair["same_choice"] == 0.5

        assert json.loads((out / "summary.json").read_text()) == summary
        table = (out / "summary.md").read_text()
        assert "| jev | jev-1.13.0 | 0.667 |" in table and "syn minus jev: -0.500" in table
        jev_rows = [json.loads(line) for line in (out / "routing" / "jev.jsonl").open()]
        assert sorted(r["example_index"] for r in jev_rows) == [0, 1, 2]
        assert "jev-key" not in (out / "routing" / "jev.jsonl").read_text()

        # Resuming reuses every scored row; only the row that failed is asked again.
        again = bench([dataset], out, targets, 2, clients={"jev": jev, "syn": syn})
    assert len(calls) == 4
    assert again["datasets"]["routing"]["targets"]["jev"] == j
    assert len((out / "routing" / "syn.jsonl").read_text().splitlines()) == 4

    # Rows from another dataset or another model are never mixed into the same file.
    write(dataset, [{**ROWS[0], "expected_option_id": "sales"}, *ROWS[1:]])
    with pytest.raises(ValueError, match="different dataset"):
        bench([dataset], out, targets[:1], clients={"jev": jev})
    write(dataset, ROWS)
    with pytest.raises(ValueError, match="another target or model"):
        bench([dataset], out, [parse_target("jev@jev-1.12.0")], clients={"jev": jev})


def test_missing_or_refused_key_stops_before_any_row(monkeypatch, tmp_path):
    dataset = tmp_path / "data.jsonl"
    write(dataset, ROWS[:2])
    calls = []
    jev = httpx.Client(transport=httpx.MockTransport(jev_handler(calls)), base_url="http://jev/")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TargetError, match="TYPESAFE_API_KEY"):
        bench([dataset], tmp_path / "out", [parse_target("jev")], clients={"jev": jev})
    assert calls == []
    monkeypatch.setenv("TYPESAFE_API_KEY", "wrong")
    with pytest.raises(TargetError, match="403"):
        bench([dataset], tmp_path / "out", [parse_target("jev")], clients={"jev": jev})
    assert len(calls) == 1 and not (tmp_path / "out" / "data" / "jev.jsonl").exists()


def test_retries_short_waits_but_not_daily_quotas(tmp_path):
    examples = [EvalExample.model_validate(row) for row in ROWS[:2]]
    answer = {"choice": "billing", "probabilities": {"sales": 0.5, "billing": 0.5}}
    ok = {"model": "m", "answers": {"answer": answer}}
    # Replies in order per state; None answers normally. The first is the warmup.
    replies = {
        "Duplicate charge": [
            None,
            httpx.Response(503),
            httpx.Response(429, headers={"retry-after": "3"}),
            None,
        ],
        "Wants a quote": [httpx.Response(429, headers={"retry-after": "3600"})],
    }
    served = []

    def handle(request):
        state = json.loads(request.content)["state"]
        served.append(state)
        return replies[state].pop(0) or httpx.Response(200, json=ok)

    waits = []
    client = httpx.Client(transport=httpx.MockTransport(handle), base_url="http://t/")
    rows = run_target(
        examples, parse_target("t=http://t"), tmp_path / "t.jsonl", client, 1, sleep=waits.append
    )
    # Warmup succeeds; then row 0 waits 1 s after the 503 and the stated 3 s after the 429.
    assert waits == [1.0, 3.0]
    assert rows[0]["choice"] == "billing" and rows[0]["latency_ms"] >= 0
    # An hour-long Retry-After is a quota: recorded once, not waited out.
    assert rows[1]["error"]["status"] == 429 and served.count("Wants a quote") == 1


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        # Names only one of the two options.
        (
            httpx.Response(
                200,
                json={"answers": {"answer": {"choice": "sales", "probabilities": {"sales": 1}}}},
            ),
            "answer probabilities do not name exactly the options",
        ),
        # A proxy's HTML page instead of an answer.
        (httpx.Response(200, text="<html>Bad gateway</html>"), "<html>Bad gateway</html>"),
    ],
)
def test_unreadable_answers_become_errors(tmp_path, reply, detail):
    examples = [EvalExample.model_validate(ROWS[0])]
    client = httpx.Client(transport=httpx.MockTransport(lambda r: reply), base_url="http://t/")
    [row] = run_target(examples, parse_target("t=http://t"), tmp_path / "t.jsonl", client, 1)
    assert row["error"] == {"status": 200, "detail": detail}


def test_cli_bench(monkeypatch, capsys):
    seen = {}

    def fake_bench(datasets, out, targets, concurrency, limit):
        seen.update(datasets=datasets, out=out, targets=targets, concurrency=concurrency)
        seen["limit"] = limit
        return {"ok": 1}

    monkeypatch.setattr("syn.bench.bench", fake_bench)
    argv = ["bench", "a.jsonl", "b.jsonl", "--out", "runs/b", "--target", "jev"]
    cli.main([*argv, "--target", "syn=https://example.test", "--limit", "5"])
    assert [str(p) for p in seen["datasets"]] == ["a.jsonl", "b.jsonl"]
    assert [t.name for t in seen["targets"]] == ["jev", "syn"]
    assert (seen["concurrency"], seen["limit"]) == (1, 5)
    assert json.loads(capsys.readouterr().out) == {"ok": 1}

    with pytest.raises(SystemExit) as exited:
        cli.main([*argv, "--target", "acme"])
    assert exited.value.code == 2

    def refused(*args):
        raise TargetError("jev needs an API key in TYPESAFE_API_KEY")

    monkeypatch.setattr("syn.bench.bench", refused)
    with pytest.raises(SystemExit, match="syn bench: jev needs an API key"):
        cli.main(argv)
