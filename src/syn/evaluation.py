import hashlib
import json
import math
import random
import time
from pathlib import Path

import httpx

from .schema import EvalExample, ScoreRequest, ScoreResponse
from .scoring import probabilities

PROFILE_FIELDS = (
    "model",
    "revision",
    "revision_commit",
    "backend",
    "prompt_version",
    "head_sha256",
)


def example_digest(example: EvalExample) -> str:
    """Identifies the row by its request and answer only, so tags added later never break resume."""
    body = {
        "request": example.request.model_dump(),
        "expected_option_id": example.expected_option_id,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def bootstrap_ci(values: list[float], samples: int = 1000, seed: int = 0) -> list[float]:
    """95% bootstrap interval of the mean, resampling examples with replacement.

    Without this, a two-point accuracy gap between two runs looks like a result when it is noise.
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    return [means[int(0.025 * samples)], means[min(int(0.975 * samples), samples - 1)]]


def expected_calibration_error(
    confidence: list[float], correct: list[int], bins: int = 10
) -> float:
    """Mean gap between stated confidence and observed accuracy, over equal-width bins."""
    n = len(confidence)
    ece = 0.0
    for bucket in range(bins):
        indices = [i for i, p in enumerate(confidence) if min(int(p * bins), bins - 1) == bucket]
        if indices:
            ece += abs(sum(confidence[i] - correct[i] for i in indices)) / n
    return ece


def paired_difference(ca: list[int], cb: list[int], samples: int = 1000, seed: int = 0) -> dict:
    """Accuracy difference b minus a on the same examples, with a paired bootstrap interval.

    Resampling examples jointly is far tighter than comparing two independent intervals.
    """
    n = len(ca)
    rng = random.Random(seed)
    diffs = sorted(
        sum(cb[j] - ca[j] for j in rng.choices(range(n), k=n)) / n for _ in range(samples)
    )
    return {
        "difference_b_minus_a": (sum(cb) - sum(ca)) / n,
        "difference_ci95": [
            diffs[int(0.025 * samples)],
            diffs[min(int(0.975 * samples), samples - 1)],
        ],
        "both_correct": sum(x and y for x, y in zip(ca, cb)),
        "only_a_correct": sum(x and not y for x, y in zip(ca, cb)),
        "only_b_correct": sum(y and not x for x, y in zip(ca, cb)),
        "neither_correct": sum(not x and not y for x, y in zip(ca, cb)),
    }


def _accepted(response: ScoreResponse, probs: list[float], best: int) -> bool:
    chance = 1.0 / len(probs)
    confidence = (probs[best] - chance) / (1.0 - chance)
    return (
        probs[best] >= response.abstain_threshold
        and confidence >= response.min_confidence
        and response.ordering_agreement >= response.min_ordering_agreement
    )


def metrics(rows: list[dict], temperature: float | None = None, bootstrap: int = 1000) -> dict:
    if not rows:
        raise ValueError("Evaluation requires at least one record")
    scored = [row for row in rows if "response" in row]
    if not scored:
        raise ValueError("No example was scored successfully")
    correct, losses, briers, confidence, accepted, latencies, agreements = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    flips, permuted_correct = [], []
    profiles = set()
    for row in scored:
        response = ScoreResponse.model_validate(row["response"])
        profiles.add(tuple(getattr(response, field) for field in PROFILE_FIELDS))
        ids = [s.id for s in response.scores]
        target = ids.index(row["expected_option_id"])
        logs = [s.log_probability for s in response.scores]
        temp = response.temperature if temperature is None else temperature
        probs = probabilities(logs, temp)
        best = max(range(len(probs)), key=probs.__getitem__)
        peak = max(logs)
        log_normalizer = math.log(sum(math.exp((x - peak) / temp) for x in logs))
        losses.append((peak - logs[target]) / temp + log_normalizer)
        briers.append(sum((p - int(i == target)) ** 2 for i, p in enumerate(probs)))
        correct.append(int(best == target))
        confidence.append(probs[best])
        accepted.append(_accepted(response, probs, best))
        agreements.append(response.ordering_agreement)
        latencies.append(row.get("client_latency_ms", response.latency_ms))
        if "permuted_response" in row:
            permuted = ScoreResponse.model_validate(row["permuted_response"])
            flips.append(permuted.best_option_id != response.best_option_id)
            permuted_correct.append(permuted.best_option_id == row["expected_option_id"])
    n = len(scored)
    ece = expected_calibration_error(confidence, correct)
    accepted_count = sum(accepted)
    profile = dict(zip(PROFILE_FIELDS, next(iter(profiles)))) if len(profiles) == 1 else None
    return {
        "examples": len(rows),
        "scored": n,
        "errors": len(rows) - n,
        "top1_accuracy": sum(correct) / n,
        "top1_accuracy_ci95": bootstrap_ci(correct, bootstrap) if bootstrap else None,
        "negative_log_likelihood": sum(losses) / n,
        "multiclass_brier": sum(briers) / n,
        "ece_10_bins": ece,
        "coverage": accepted_count / n,
        "accepted_accuracy": sum(c for c, a in zip(correct, accepted) if a) / accepted_count
        if accepted_count
        else None,
        "mean_ordering_agreement": sum(agreements) / n,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "option_order_flip_rate": sum(flips) / len(flips) if flips else None,
        "permuted_top1_accuracy": sum(permuted_correct) / len(permuted_correct)
        if permuted_correct
        else None,
        "profile": profile,
    }


def _score_into(client: httpx.Client, request: ScoreRequest, row: dict, key: str) -> None:
    """Record either `<key>` or `<key>_error`; a failing example must not stop the run."""
    started = time.perf_counter()
    try:
        result = client.post("v1/score", json=request.model_dump())
    except httpx.HTTPError as exc:
        row[f"{key}_error"] = {"status": None, "detail": str(exc)[:500]}
        return
    if result.status_code != 200:
        row[f"{key}_error"] = {"status": result.status_code, "detail": result.text[:500]}
        return
    row[key] = ScoreResponse.model_validate(result.json()).model_dump()
    if key == "response":
        row["client_latency_ms"] = (time.perf_counter() - started) * 1000


def evaluate(
    dataset: Path,
    output: Path,
    url: str,
    shuffle: bool,
    seed: int,
    client: httpx.Client | None = None,
) -> dict:
    # Validate the complete dataset before contacting a model or creating output.
    examples = [EvalExample.model_validate(row) for row in read_jsonl(dataset)]
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    # An existing output file is resumed, never overwritten.
    previous = {}
    if output.exists() and output.stat().st_size:
        for row in read_jsonl(output):
            previous[row["example_index"]] = row
    own_client = client is None
    client = client or httpx.Client(base_url=url.rstrip("/") + "/", timeout=180)
    rows = []
    try:
        client.get("health").raise_for_status()
        with output.open("a") as out:
            for index, example in enumerate(examples):
                permuted_options = None
                if shuffle:
                    # Consume the generator even for resumed rows so the permutation is stable.
                    permuted_options = list(example.request.options)
                    rng.shuffle(permuted_options)
                    if permuted_options == example.request.options:
                        permuted_options = permuted_options[1:] + permuted_options[:1]
                digest = example_digest(example)
                if index in previous:
                    if previous[index].get("example_sha256") != digest:
                        raise ValueError(
                            f"{output} was produced from a different dataset (example {index})"
                        )
                    rows.append(previous[index])
                    continue
                row = {"example_index": index, "example_sha256": digest, **example.model_dump()}
                _score_into(client, example.request, row, "response")
                if permuted_options is not None and "response" in row:
                    permuted = example.request.model_copy(update={"options": permuted_options})
                    _score_into(client, permuted, row, "permuted_response")
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
                rows.append(row)
    finally:
        if own_client:
            client.close()
    return metrics(rows)


def fit_temperature(path: Path) -> dict:
    rows = [row for row in read_jsonl(path) if "response" in row]
    if not rows:
        raise ValueError("No scored examples to fit on")
    profiles = {tuple(r["response"].get(field) for field in PROFILE_FIELDS) for r in rows}
    if len(profiles) != 1:
        raise ValueError("Fit temperature on one model/revision/backend/prompt profile at a time")
    # A bounded log-spaced search is sufficient for this one-parameter baseline.
    candidates = sorted(
        {1.0, *[math.exp(math.log(0.05) + i * math.log(400) / 200) for i in range(201)]}
    )
    nll = lambda t: metrics(rows, t, bootstrap=0)["negative_log_likelihood"]
    best = min(candidates, key=nll)
    return {
        **dict(zip(PROFILE_FIELDS, next(iter(profiles)))),
        "temperature": best,
        "examples": len(rows),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fit_nll_before": nll(1.0),
        "fit_nll_after": nll(best),
        "at_search_boundary": best in (candidates[0], candidates[-1]),
        "note": "Fit-set metrics only. Evaluate on a separate test split before deployment.",
    }


def _correct(row: dict) -> int:
    return int(row["response"]["best_option_id"] == row["expected_option_id"])


def compare(path_a: Path, path_b: Path, samples: int = 1000, seed: int = 0) -> dict:
    """Paired comparison of two evaluation outputs over the same dataset.

    Rows are matched by example index and verified by the example digest, so the two runs must come
    from the same JSONL. The interval is a paired bootstrap of the accuracy difference, which is far
    tighter than comparing two independent intervals.
    """
    a = {r["example_index"]: r for r in read_jsonl(path_a) if "response" in r}
    b = {r["example_index"]: r for r in read_jsonl(path_b) if "response" in r}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("The two runs share no successfully scored examples")
    for index in common:
        if a[index].get("example_sha256") != b[index].get("example_sha256"):
            raise ValueError(f"Example {index} differs between the runs; same dataset required")
    ca = [_correct(a[i]) for i in common]
    cb = [_correct(b[i]) for i in common]
    n = len(common)
    profile = lambda rows: dict(
        zip(PROFILE_FIELDS, (rows[common[0]]["response"].get(f) for f in PROFILE_FIELDS))
    )
    return {
        "examples": n,
        "accuracy_a": sum(ca) / n,
        "accuracy_b": sum(cb) / n,
        **paired_difference(ca, cb, samples, seed),
        "profile_a": profile(a),
        "profile_b": profile(b),
        "bootstrap_samples": samples,
    }
