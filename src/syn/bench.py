"""Benchmark System One endpoints, such as TypeSafe's Jev and this service, on the same rows.

    syn bench data/eval/agnews-250.jsonl --out runs/bench --target jev --target syn=http://127.0.0.1:8765

Every labeled row becomes one System One request holding a single choice question: the row's
context is the state, its question the instructions, its options the criteria. Every target
receives the same request body, so differences come from the models, not the prompts.

A target is `NAME[@MODEL][=URL]`. `jev` defaults to https://api.typesafe.ai and jev-latest and
needs TYPESAFE_API_KEY. Any other name needs a URL, asks for syn-latest, and sends
`<NAME>_API_KEY` as a bearer token when that variable is set. The first target is the
reference every other target is paired against.
"""

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx

from .evaluation import (
    bootstrap_ci,
    expected_calibration_error,
    paired_difference,
    percentile,
    read_jsonl,
)
from .schema import EvalExample, ScoreRequest

QUESTION = "answer"
RETRY_STATUSES = {429, 500, 502, 503, 504}
# A target that is missing, unreachable, or refuses the key fails the run instead of every row.
FATAL_STATUSES = {None, 401, 403, 404}
# Longer waits are daily quotas, not congestion; retrying would stall the run for hours.
MAX_RETRY_AFTER_S = 60.0
# Log loss clips here, so one confident miss costs about 13.8 nats instead of infinity.
PROBABILITY_FLOOR = 1e-6
SPLIT_NAMES = {"train", "validation", "test"}


class TargetError(RuntimeError):
    """A target cannot serve the run at all: no key, a rejected key, or no endpoint."""


@dataclass(frozen=True)
class Target:
    name: str
    url: str
    model: str
    key_env: str
    key_required: bool = False

    def headers(self) -> dict[str, str]:
        key = os.environ.get(self.key_env)
        return {"Authorization": f"Bearer {key}"} if key else {}


PRESETS = {
    "jev": Target("jev", "https://api.typesafe.ai", "jev-latest", "TYPESAFE_API_KEY", True),
}


def parse_target(spec: str) -> Target:
    head, _, url = spec.partition("=")
    name, _, model = head.partition("@")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(f"Target name {name!r} must be lowercase letters, digits, - or _")
    preset = PRESETS.get(name)
    if preset is None and not url:
        raise ValueError(f"Target {name!r} needs a URL, e.g. {name}=http://127.0.0.1:8765")
    return Target(
        name=name,
        url=(url or preset.url).rstrip("/"),
        model=model or (preset.model if preset else "syn-latest"),
        key_env=preset.key_env if preset else f"{name.upper().replace('-', '_')}_API_KEY",
        key_required=preset.key_required if preset else False,
    )


def to_systemone(request: ScoreRequest, model: str) -> dict:
    """One choice question; an option whose text is only its id goes without a description."""
    instructions = request.question
    if request.criteria:
        instructions += "\n\n" + request.criteria
    return {
        "state": request.context,
        "model": model,
        "questions": {
            QUESTION: {
                "type": "choice",
                "instructions": instructions,
                "criteria": {o.id: None if o.text == o.id else o.text for o in request.options},
            }
        },
    }


def _retry_after(response: httpx.Response) -> float | None:
    for header, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            return float(response.headers[header]) / scale
        except (KeyError, ValueError):
            continue
    return None


def _post(
    client: httpx.Client, target: Target, body: dict, retries: int, sleep: Callable
) -> tuple[dict | None, dict | None, float]:
    """(response body, error, latency ms of the last attempt). Retries 429, 5xx and timeouts."""
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            result = client.post("v1/systemone", json=body, headers=target.headers())
        except httpx.HTTPError as exc:
            latency = (time.perf_counter() - started) * 1000
            error, wait = {"status": None, "detail": str(exc)[:500]}, 2.0**attempt
        else:
            latency = (time.perf_counter() - started) * 1000
            if result.status_code == 200:
                try:
                    return result.json(), None, latency
                except ValueError:
                    return None, {"status": 200, "detail": result.text[:500]}, latency
            error = {"status": result.status_code, "detail": result.text[:500]}
            if result.status_code not in RETRY_STATUSES:
                return None, error, latency
            wait = _retry_after(result) or 2.0**attempt
        if attempt == retries or wait > MAX_RETRY_AFTER_S:
            break
        sleep(wait)
    return None, error, latency


def _answer(body: dict, request: ScoreRequest) -> dict:
    """The recorded fields of a choice answer; raises ValueError when it cannot be scored."""
    try:
        answer = body["answers"][QUESTION]
        probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
        choice = str(answer["choice"])
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"no readable choice answer: {exc!r}") from exc
    if set(probabilities) != {o.id for o in request.options} or choice not in probabilities:
        raise ValueError("answer probabilities do not name exactly the options")
    values = probabilities.values()
    if not all(math.isfinite(p) and p >= 0 for p in values) or sum(values) <= 0:
        raise ValueError("answer probabilities are not a distribution")
    usage = body.get("usage")
    return {
        "model": body.get("model"),
        "choice": choice,
        "probabilities": probabilities,
        "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
    }


def _digest(example: EvalExample) -> str:
    # Same digest as `syn evaluate`, so rows from both commands identify examples alike.
    return hashlib.sha256(json.dumps(example.model_dump(), sort_keys=True).encode()).hexdigest()


def run_target(
    examples: list[EvalExample],
    target: Target,
    output: Path,
    client: httpx.Client,
    concurrency: int = 1,
    retries: int = 4,
    sleep: Callable = time.sleep,
) -> list[dict]:
    """Score every example on one target; resumes from `output`, retrying rows that failed."""
    digests = [_digest(e) for e in examples]
    done = {}
    if output.exists() and output.stat().st_size:
        for row in read_jsonl(output):
            index = row["example_index"]
            if index >= len(examples):  # beyond --limit
                continue
            if row["example_sha256"] != digests[index]:
                raise ValueError(
                    f"{output} was produced from a different dataset (example {index})"
                )
            if (row["target_url"], row["requested_model"]) != (target.url, target.model):
                raise ValueError(f"{output} was produced by another target or model")
            if "probabilities" in row:
                done[index] = row
    pending = [i for i in range(len(examples)) if i not in done]
    if not pending:
        return [done[i] for i in range(len(examples))]
    if target.key_required and not os.environ.get(target.key_env):
        raise TargetError(f"{target.name} needs an API key in {target.key_env}")

    # One unrecorded request first: it wakes a scaled-to-zero GPU so the cold start is not
    # timed, and a missing endpoint or refused key stops the run before any row is written.
    warmup = to_systemone(examples[pending[0]].request, target.model)
    _, error, _ = _post(client, target, warmup, retries, sleep)
    if error and error["status"] in FATAL_STATUSES:
        raise TargetError(f"{target.name} at {target.url}: {error['status']} {error['detail']}")

    def score(index: int) -> dict:
        example = examples[index]
        row = {
            "example_index": index,
            "example_sha256": digests[index],
            "expected_option_id": example.expected_option_id,
            "target": target.name,
            "target_url": target.url,
            "requested_model": target.model,
        }
        body, error, latency = _post(
            client, target, to_systemone(example.request, target.model), retries, sleep
        )
        if body is not None:
            try:
                return {**row, **_answer(body, example.request), "latency_ms": latency}
            except ValueError as exc:
                error = {"status": 200, "detail": str(exc)[:500]}
        return {**row, "error": error}

    output.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max(1, concurrency))
    try:
        with output.open("a") as out:
            for future in as_completed([pool.submit(score, i) for i in pending]):
                row = future.result()
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
                done[row["example_index"]] = row
    finally:
        # On Ctrl-C, drop the queued requests instead of sending them all before exiting.
        pool.shutdown(cancel_futures=True)
    return [done[i] for i in range(len(examples))]


def _correct(row: dict) -> int:
    return int(row["choice"] == row["expected_option_id"])


def target_metrics(rows: list[dict], bootstrap: int = 1000) -> dict:
    scored = [row for row in rows if "probabilities" in row]
    statuses = {}
    for row in rows:
        if "error" in row:
            status = str(row["error"]["status"])
            statuses[status] = statuses.get(status, 0) + 1
    result = {"examples": len(rows), "scored": len(scored), "errors": len(rows) - len(scored)}
    result["error_statuses"] = statuses
    if not scored:
        return result
    correct, confidence, losses, briers = [], [], [], []
    for row in scored:
        total = sum(row["probabilities"].values())
        probs = {k: v / total for k, v in row["probabilities"].items()}
        expected = row["expected_option_id"]
        correct.append(_correct(row))
        confidence.append(probs[row["choice"]])
        losses.append(-math.log(max(probs[expected], PROBABILITY_FLOOR)))
        briers.append(sum((p - (k == expected)) ** 2 for k, p in probs.items()))
    latencies = [row["latency_ms"] for row in scored]
    tokens = [row["input_tokens"] for row in scored if row.get("input_tokens") is not None]
    n = len(scored)
    return {
        **result,
        "models": sorted({row["model"] for row in scored if row.get("model")}),
        "top1_accuracy": sum(correct) / n,
        "top1_accuracy_ci95": bootstrap_ci(correct, bootstrap) if bootstrap else None,
        "negative_log_likelihood": sum(losses) / n,
        "multiclass_brier": sum(briers) / n,
        "ece_10_bins": expected_calibration_error(confidence, correct),
        "mean_confidence": sum(confidence) / n,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "input_tokens": sum(tokens) if tokens else None,
    }


def paired(reference: list[dict], other: list[dict], samples: int = 1000) -> dict | None:
    """`other` against `reference` on the examples both scored; a is the reference."""
    common = [
        i for i, (a, b) in enumerate(zip(reference, other)) if "choice" in a and "choice" in b
    ]
    if not common:
        return None
    ca = [_correct(reference[i]) for i in common]
    cb = [_correct(other[i]) for i in common]
    return {
        "examples": len(common),
        "accuracy_reference": sum(ca) / len(common),
        "accuracy": sum(cb) / len(common),
        **paired_difference(ca, cb, samples),
        "same_choice": sum(reference[i]["choice"] == other[i]["choice"] for i in common)
        / len(common),
    }


def dataset_name(path: Path) -> str:
    """`agnews-250` for data/eval/agnews-250.jsonl, `emotion-test` for data/emotion/test.jsonl."""
    return f"{path.parent.name}-{path.stem}" if path.stem in SPLIT_NAMES else path.stem


def markdown(summary: dict) -> str:
    lines = [f"Latency is client round trip at concurrency {summary['concurrency']}.", ""]
    for name, result in summary["datasets"].items():
        lines += [
            f"## {name} ({result['examples']} examples)",
            "",
            "| target | model | accuracy | 95% CI | NLL | Brier | ECE | p50 ms | p95 ms | errors |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for target, m in result["targets"].items():
            if not m["scored"]:
                lines.append(f"| {target} | | | | | | | | | {m['errors']} |")
                continue
            low, high = m["top1_accuracy_ci95"] or (math.nan, math.nan)
            lines.append(
                f"| {target} | {', '.join(m['models'])} | {m['top1_accuracy']:.3f} "
                f"| {low:.3f}-{high:.3f} | {m['negative_log_likelihood']:.3f} "
                f"| {m['multiclass_brier']:.3f} | {m['ece_10_bins']:.3f} "
                f"| {m['latency_p50_ms']:.0f} | {m['latency_p95_ms']:.0f} | {m['errors']} |"
            )
        for target, p in result["paired_vs_reference"].items():
            if p is None:
                continue
            low, high = p["difference_ci95"]
            lines.append(
                f"\n{target} minus {summary['reference']}: {p['difference_b_minus_a']:+.3f} "
                f"[{low:+.3f}, {high:+.3f}] over {p['examples']} paired examples; "
                f"same choice {p['same_choice']:.0%}"
            )
        lines.append("")
    return "\n".join(lines)


def bench(
    datasets: list[Path],
    out: Path,
    targets: list[Target],
    concurrency: int = 1,
    limit: int = 0,
    retries: int = 4,
    clients: dict[str, httpx.Client] | None = None,
    sleep: Callable = time.sleep,
) -> dict:
    if not targets or len({t.name for t in targets}) != len(targets):
        raise ValueError("Give one or more targets with distinct names")
    # Validate every dataset before contacting any target.
    loaded = {}
    for path in datasets:
        examples = [EvalExample.model_validate(row) for row in read_jsonl(path)]
        name = dataset_name(path)
        if name in loaded:
            raise ValueError(f"Two datasets are both named {name!r}; rename one")
        loaded[name] = examples[:limit] if limit else examples
    reference = targets[0].name
    summary = {"reference": reference, "concurrency": concurrency, "datasets": {}}
    for name, examples in loaded.items():
        runs = {}
        for target in targets:
            client = (clients or {}).get(target.name)
            own = client is None
            client = client or httpx.Client(base_url=target.url + "/", timeout=180)
            try:
                path = out / name / f"{target.name}.jsonl"
                runs[target.name] = run_target(
                    examples, target, path, client, concurrency, retries, sleep
                )
            finally:
                if own:
                    client.close()
        summary["datasets"][name] = {
            "examples": len(examples),
            "targets": {t: target_metrics(rows) for t, rows in runs.items()},
            "paired_vs_reference": {
                t: paired(runs[reference], rows) for t, rows in runs.items() if t != reference
            },
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "summary.md").write_text(markdown(summary))
    return summary
