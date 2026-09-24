"""Zero-shot image classification accuracy of the letters readout, on Imagenette.

Imagenette is ten easily told-apart ImageNet classes. Each sampled image is scored against all
ten class names as labels, exactly as `POST /` with an `image` does, and the report gives
accuracy, a per-class breakdown, and latency. It is a sanity check that images work end to end,
not a hard benchmark.

    HF_HOME=.cache/huggingface uv run --extra local --extra hub python scripts/image_benchmark.py \\
        --per-class 20 --out runs/image-bench.json
"""

import argparse
import base64
import io
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

DATASET = "johnowhitaker/imagenette2-320"
# The dataset's WordNet ids, in label order, and the class names used as labels.
CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def data_uri(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def sample(per_class: int, seed: int):
    """`per_class` images of each class, drawn reproducibly from a shuffled stream."""
    from datasets import load_dataset

    names = list(CLASSES.values())
    stream = load_dataset(DATASET, split="train", streaming=True).shuffle(
        seed=seed, buffer_size=2000
    )
    taken: dict[str, list] = defaultdict(list)
    for row in stream:
        name = names[row["label"]]
        if len(taken[name]) < per_class:
            taken[name].append(row["image"])
        if all(len(taken[n]) >= per_class for n in names):
            break
    return [(name, image) for name in names for image in taken[name]]


def build_scorer():
    from transformers import AutoTokenizer

    from syn.api import load_processor
    from syn.artifacts import pretrained_call
    from syn.backends import LocalBackend, resolve_config, revision_commit
    from syn.config import Settings
    from syn.prompt import PromptBuilder
    from syn.scoring import Scorer

    settings = Settings(images=True)
    config = resolve_config(settings)
    model, extra = pretrained_call(settings.model, settings.revision)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False, **extra)
    builder = PromptBuilder(
        tokenizer,
        settings.max_prompt_tokens,
        settings.prompt_format,
        load_processor(settings, model, extra, tokenizer),
        settings.image_max_pixels,
    )
    return settings, Scorer(
        settings, builder, LocalBackend(settings, config), revision_commit(config)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from syn.schema import ScoreRequest
    from syn.shorthand import build_request

    settings, scorer = build_scorer()
    labels = list(CLASSES.values())
    # Shuffled once, so a label's letter says nothing about the answer.
    random.Random(args.seed).shuffle(labels)
    rows = sample(args.per_class, args.seed)
    scorer.score(ScoreRequest.model_validate(build_request(labels, "", None, data_uri(rows[0][1]))))

    results, latencies = [], []
    by_class: dict[str, list[bool]] = defaultdict(list)
    for truth, image in rows:
        started = time.perf_counter()
        response = scorer.score(
            ScoreRequest.model_validate(build_request(labels, "", None, data_uri(image)))
        )
        latencies.append((time.perf_counter() - started) * 1000)
        correct = response.best_option_id == truth
        by_class[truth].append(correct)
        best = max(response.scores, key=lambda s: s.probability)
        results.append({"truth": truth, "predicted": best.id, "probability": best.probability})
    accuracy = sum(r["truth"] == r["predicted"] for r in results) / len(results)
    report = {
        "model": settings.model,
        "dataset": DATASET,
        "images": len(results),
        "accuracy": round(accuracy, 4),
        "chance": round(1 / len(labels), 4),
        "per_class": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_class.items())},
        "latency_ms": {
            "median": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
        },
        "image_max_pixels": settings.image_max_pixels,
        "device": scorer.backend.device,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**report, "results": results}, indent=2))


if __name__ == "__main__":
    main()
