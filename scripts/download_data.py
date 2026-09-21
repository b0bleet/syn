"""Rebuild everything under data/ from Hugging Face and the synthetic generator.

    uv run --extra hub python scripts/download_data.py

Reproduces the exact files used for the reported numbers: five imported datasets with the
same sizes, questions, and seeds as each meta.json, the synthetic routing set, the balanced
sms_spam spam-eval file, and the 250-row eval subsets. Idempotent: targets that already exist
are skipped, so re-running only fills in what is missing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from syn.synthetic import generate

DATA = Path("data")
SEED = 7

# (dataset id, output dir, convert() kwargs) matching each data/*/meta.json.
IMPORTS = [
    (
        "fancyzhx/ag_news",
        "agnews",
        {"train": 4000, "validation": 500, "test": 1000, "all_options": True},
    ),
    (
        "legacy-datasets/banking77",
        "banking77",
        {"train": 4000, "validation": 500, "test": 1000, "min_options": 3, "max_options": 8},
    ),
    ("fancyzhx/dbpedia_14", "dbpedia", {"train": 4000, "validation": 500, "test": 500}),
    (
        "cornell-movie-review-data/rotten_tomatoes",
        "rotten",
        {"train": 4000, "validation": 500, "test": 500},
    ),
    ("dair-ai/emotion", "emotion", {"train": 0, "validation": 0, "test": 500}),
]


def build_spam_eval(path: Path, spam_n: int = 100, ham_n: int = 100) -> None:
    """Balanced sms_spam sample in the spam/not spam shorthand wording."""
    from datasets import load_dataset

    ds = load_dataset("ucirvine/sms_spam")["train"]
    spam = [r["sms"].strip() for r in ds if r["label"] == 1]
    ham = [r["sms"].strip() for r in ds if r["label"] == 0]
    rng = random.Random(SEED)
    rng.shuffle(spam)
    rng.shuffle(ham)
    rows = [
        {
            "request": {
                "context": text,
                "question": "Which label best applies to the text?",
                "options": [
                    {"id": "spam", "text": "spam"},
                    {"id": "not spam", "text": "not spam"},
                ],
            },
            "expected_option_id": label,
        }
        for text, label in [(t, "spam") for t in spam[:spam_n]]
        + [(t, "not spam") for t in ham[:ham_n]]
    ]
    rng.shuffle(rows)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    from syn.hub import convert

    for dataset, name, kwargs in IMPORTS:
        out = DATA / name
        if (out / "test.jsonl").exists():
            print(f"{name}: already present, skipping")
            continue
        meta = convert(dataset, out, seed=SEED, **kwargs)
        print(f"{name}: {meta['train']}+{meta['validation']}+{meta['test']} rows")

    synthetic = DATA / "synthetic"
    if (synthetic / "test.jsonl").exists():
        print("synthetic: already present, skipping")
    else:
        meta = generate(synthetic, seed=SEED)
        print(f"synthetic: {meta['train']}+{meta['validation']}+{meta['test']} rows")

    spam_eval = DATA / "spam-eval.jsonl"
    if spam_eval.exists():
        print("spam-eval: already present, skipping")
    else:
        build_spam_eval(spam_eval)
        print("spam-eval: 200 rows")

    rng = random.Random(SEED)
    for name in ("agnews", "banking77", "synthetic"):
        src, dst = DATA / name / "test.jsonl", DATA / "eval" / f"{name}-250.jsonl"
        if dst.exists():
            print(f"eval/{dst.name}: already present, skipping")
            continue
        if not src.exists():
            print(f"eval/{dst.name}: {src} missing, skipping")
            continue
        rows = [json.loads(line) for line in src.open()]
        rng.shuffle(rows)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:250]))
        print(f"eval/{dst.name}: 250 rows")


if __name__ == "__main__":
    main()
