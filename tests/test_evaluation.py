import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient
from helpers import CharacterTokenizer, ContentBackend, billing_backend, make_scorer

from syn.api import create_app
from syn.evaluation import bootstrap_ci, compare, evaluate, fit_temperature, metrics
from syn.schema import ScoreRequest

BASE = {
    "question": "Choose a team",
    "options": [{"id": "sales", "text": "Sales"}, {"id": "billing", "text": "Billing"}],
}


def dataset_rows():
    return [
        {"request": {"context": "Duplicate charge", **BASE}, "expected_option_id": "billing"},
        {"request": {"context": "Wants a quote", **BASE}, "expected_option_id": "sales"},
        # Longer than the default token limit for the one-token-per-character fake tokenizer,
        # so the service rejects it with 422 and the run must record the error and continue.
        {"request": {"context": "x" * 9000, **BASE}, "expected_option_id": "billing"},
    ]


def write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


@pytest.fixture
def app():
    tokenizer = CharacterTokenizer()
    return create_app(scorer=make_scorer(backend=billing_backend(tokenizer), tokenizer=tokenizer))


def test_evaluate_records_errors_shuffles_and_resumes(app, tmp_path):
    dataset, output = tmp_path / "data.jsonl", tmp_path / "out.jsonl"
    write(dataset, dataset_rows())
    with TestClient(app) as client:
        result = evaluate(dataset, output, "unused", shuffle=True, seed=1, client=client)
    assert (result["examples"], result["scored"], result["errors"]) == (3, 2, 1)
    assert result["top1_accuracy"] == 0.5
    low, high = result["top1_accuracy_ci95"]
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert result["option_order_flip_rate"] == 0.0
    assert result["mean_ordering_agreement"] == 1.0
    assert result["profile"]["revision_commit"] == "abc123"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["example_index"] for row in rows] == [0, 1, 2]
    assert rows[2]["response_error"]["status"] == 422 and "response" not in rows[2]
    assert "permuted_response" in rows[0] and "client_latency_ms" in rows[0]

    # Resuming reuses every stored row and does not grow the file.
    with TestClient(app) as client:
        again = evaluate(dataset, output, "unused", shuffle=True, seed=1, client=client)
    assert again == result
    assert len(output.read_text().splitlines()) == 3

    # A different dataset may not be appended to the same output.
    changed = dataset_rows()
    changed[0]["expected_option_id"] = "sales"
    write(dataset, changed)
    with TestClient(app) as client, pytest.raises(ValueError, match="different dataset"):
        evaluate(dataset, output, "unused", shuffle=False, seed=1, client=client)


def test_evaluate_fails_fast_when_service_is_down(tmp_path):
    dataset, output = tmp_path / "data.jsonl", tmp_path / "out.jsonl"
    write(dataset, dataset_rows())
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    with (
        httpx.Client(transport=transport, base_url="http://down/") as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        evaluate(dataset, output, "unused", shuffle=False, seed=1, client=client)
    assert not output.exists()


def test_metrics_and_calibration(request_data, tmp_path):
    tokenizer = CharacterTokenizer()
    scorer = make_scorer(backend=billing_backend(tokenizer), tokenizer=tokenizer)
    response = scorer.score(ScoreRequest.model_validate(request_data)).model_dump()
    rows = [{"response": response, "expected_option_id": t} for t in ["billing", "sales"]]
    rows.append({"expected_option_id": "billing", "response_error": {"status": 422, "detail": ""}})
    result = metrics(rows)
    assert (result["examples"], result["scored"], result["errors"]) == (3, 2, 1)
    assert result["top1_accuracy"] == 0.5
    assert result["negative_log_likelihood"] == pytest.approx(-math.log(0.75 * 0.25) / 2)
    assert result["multiclass_brier"] == pytest.approx(0.625)
    assert result["ece_10_bins"] == pytest.approx(0.25)
    assert result["coverage"] == 1.0
    rows[0]["permuted_response"] = {**response, "best_option_id": "sales"}
    shuffled = metrics(rows)
    assert shuffled["option_order_flip_rate"] == 1
    assert shuffled["permuted_top1_accuracy"] == 0

    path = tmp_path / "calibration.jsonl"
    write(path, rows)
    fitted = fit_temperature(path)
    assert fitted["temperature"] > 1
    assert fitted["fit_nll_after"] < fitted["fit_nll_before"]
    assert fitted["revision_commit"] == "abc123" and fitted["examples"] == 2
    rows[1]["response"] = {**response, "model": "different-model"}
    write(path, rows)
    with pytest.raises(ValueError, match="one model"):
        fit_temperature(path)
    with pytest.raises(ValueError, match="No example"):
        metrics([{"response_error": {}}])
    assert metrics(rows, bootstrap=0)["top1_accuracy_ci95"] is None


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [1] * 80 + [0] * 20
    low, high = bootstrap_ci(values, samples=500, seed=1)
    assert low < 0.8 < high and 0.7 < low and high < 0.9
    assert bootstrap_ci(values, samples=500, seed=1) == [low, high]
    assert bootstrap_ci([1, 1, 1, 1], samples=50) == [1.0, 1.0]


def test_compare_pairs_runs_on_the_same_dataset(app, tmp_path):
    dataset = tmp_path / "data.jsonl"
    write(dataset, dataset_rows()[:2])
    run_a, run_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with TestClient(app) as client:  # billing always wins: 1 of 2 correct
        evaluate(dataset, run_a, "unused", shuffle=False, seed=1, client=client)
    tokenizer = CharacterTokenizer()
    sales_backend = ContentBackend(tokenizer, {"Sales": math.log(0.6), "Billing": math.log(0.2)})
    other = create_app(scorer=make_scorer(backend=sales_backend, tokenizer=tokenizer))
    with TestClient(other) as client:  # sales always wins: the other 1 of 2 correct
        evaluate(dataset, run_b, "unused", shuffle=False, seed=1, client=client)
    result = compare(run_a, run_b, samples=200)
    assert result["examples"] == 2
    assert result["accuracy_a"] == 0.5 and result["accuracy_b"] == 0.5
    assert result["difference_b_minus_a"] == 0.0
    assert result["only_a_correct"] == 1 and result["only_b_correct"] == 1
    assert result["both_correct"] == 0 and result["neither_correct"] == 0
    low, high = result["difference_ci95"]
    assert low <= 0.0 <= high
    assert result["profile_a"]["prompt_version"] == "qwen-options-v1"
    # A run on a different dataset is refused.
    changed = dataset_rows()[:2]
    changed[0]["expected_option_id"] = "sales"
    write(dataset, changed)
    run_c = tmp_path / "c.jsonl"
    with TestClient(app) as client:
        evaluate(dataset, run_c, "unused", shuffle=False, seed=1, client=client)
    with pytest.raises(ValueError, match="differs between the runs"):
        compare(run_a, run_c)
