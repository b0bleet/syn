"""Convert a public Hugging Face classification dataset into decision rows.

A classification row is already a decision: the text is the state, the classes are the options,
and the existing label is the answer. The only real work is choosing which options to show. This
follows kev's approach of reshaping public datasets rather than waiting for private labels.

    syn import-hf fancyzhx/ag_news --out data/agnews

Two design choices worth knowing:

* Options are sampled, not always the full class list. The correct class plus a few distractors
  keeps rows short when a dataset has 77 classes, and it forces the model to read the options
  rather than memorise a fixed list. `--all-options` keeps every class instead, which is the
  honest setting when comparing against a published classification benchmark.
* A held-out split is carved from train when the dataset has no validation split, so test is
  never touched during training or model selection.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from .schema import EvalExample

# Mirrors the option limits in schema.ScoreRequest. Every emitted row is also validated against
# the schema itself, so a drift here fails loudly at conversion time rather than at serving time.
MIN_OPTIONS, MAX_OPTIONS = 2, 26
# Datasets whose class names are terse identifiers read better as spaced words.
UNDERSCORE = re.compile(r"[_\-]+")
DEFAULT_QUESTION = "Which label best applies to the text?"
QUESTIONS = {
    "fancyzhx/ag_news": "What is the topic of this news article?",
    "legacy-datasets/banking77": "Which banking intent best describes this customer message?",
    "dair-ai/emotion": "Which emotion does this text express?",
    "SetFit/sst5": "What is the sentiment of this review?",
    "stanfordnlp/imdb": "What is the sentiment of this review?",
}
# Optional gloss per class, so options carry meaning rather than a bare word.
GLOSSES = {
    "fancyzhx/ag_news": {
        "World": "World news: politics, international affairs, conflicts",
        "Sports": "Sports: games, athletes, teams, results",
        "Business": "Business: companies, markets, economy, finance",
        "Sci/Tech": "Science and technology: research, computing, space",
    },
}


def label_text(dataset: str, name: str) -> str:
    gloss = GLOSSES.get(dataset, {}).get(name)
    return gloss if gloss else UNDERSCORE.sub(" ", name).strip()


def _columns(features) -> tuple[str, str, list[str]]:
    """Find the text column, the label column, and the class names."""
    label = next(
        (c for c in ("label", "labels", "intent", "category", "class") if c in features), None
    )
    if label is None:
        label = next((c for c in features if "label" in c.lower()), None)
    if label is None:
        raise ValueError(f"No label column found in {list(features)}")
    names = getattr(features[label], "names", None)
    if not names:
        raise ValueError(
            f"Column {label!r} has no class names; only ClassLabel datasets are supported"
        )
    text = next(
        (c for c in ("text", "sentence", "content", "sms", "query", "review") if c in features),
        None,
    )
    if text is None:
        text = next((c for c in features if c != label), None)
    if text is None:
        raise ValueError(f"No text column found in {list(features)}")
    return text, label, list(names)


def make_row(
    dataset: str,
    question: str,
    text: str,
    correct: str,
    names: list[str],
    rng: random.Random,
    min_options: int,
    max_options: int,
    all_options: bool,
) -> dict:
    if all_options:
        chosen = list(names)
    else:
        count = rng.randint(min_options, min(max_options, len(names)))
        pool = [n for n in names if n != correct]
        chosen = [correct, *rng.sample(pool, min(count - 1, len(pool)))]
    rng.shuffle(chosen)
    return {
        "request": {
            "context": text,
            "question": question,
            "options": [{"id": n, "text": label_text(dataset, n)} for n in chosen],
        },
        "expected_option_id": correct,
    }


def convert(
    dataset: str,
    out: Path,
    train: int = 4000,
    validation: int = 500,
    test: int = 1000,
    seed: int = 7,
    min_options: int = 2,
    max_options: int = 6,
    all_options: bool = False,
    question: str | None = None,
    max_chars: int = 2000,
    revision: str | None = None,
) -> dict:
    """Write train/validation/test JSONL in this project's row format."""
    from datasets import load_dataset

    if not MIN_OPTIONS <= min_options <= max_options <= MAX_OPTIONS:
        raise ValueError(
            f"Need {MIN_OPTIONS} <= min_options <= max_options <= {MAX_OPTIONS}; "
            f"got {min_options} and {max_options}"
        )
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        if (out / f"{split}.jsonl").exists():
            raise FileExistsError(out / f"{split}.jsonl")

    loaded = load_dataset(dataset, revision=revision)
    available = set(loaded)
    if "train" not in available:
        raise ValueError(f"{dataset} has no train split; found {sorted(available)}")
    text_col, label_col, names = _columns(loaded["train"].features)
    if all_options and len(names) > MAX_OPTIONS:
        raise ValueError(
            f"{dataset} has {len(names)} classes but a request holds at most {MAX_OPTIONS} "
            "options. Drop --all-options to sample distractors instead."
        )
    question = question or QUESTIONS.get(dataset, DEFAULT_QUESTION)

    # Carve validation from train when the dataset has none, so test stays untouched.
    test_split = "test" if "test" in available else "validation"
    if test_split not in available:
        raise ValueError(f"{dataset} has no test or validation split; found {sorted(available)}")
    train_pool = loaded["train"].shuffle(seed=seed)
    if "validation" in available and test_split != "validation":
        val_pool = loaded["validation"].shuffle(seed=seed)
        train_rows, val_rows = train_pool, val_pool
    else:
        held = min(validation * 4, len(train_pool) // 4)
        val_rows, train_rows = (
            train_pool.select(range(held)),
            train_pool.select(range(held, len(train_pool))),
        )
    pools = {
        "train": (train_rows, train),
        "validation": (val_rows, validation),
        "test": (loaded[test_split].shuffle(seed=seed + 1), test),
    }

    counts, skipped, leaked = {}, 0, 0
    # One registry across all splits, keyed by normalised text and valued by the split that
    # claimed it. Public datasets contain cross-split duplicates; a text that went into train
    # must never reach test, or the test score is partly memorisation.
    seen: dict[str, str] = {}
    for split, (pool, wanted) in pools.items():
        rng = random.Random(hashlib.sha256(f"{seed}:{split}".encode()).hexdigest()[:16])
        rows = []
        for example in pool:
            if len(rows) >= wanted:
                break
            body = example[text_col]
            if not isinstance(body, str):
                skipped += 1
                continue
            body = " ".join(body.split())[:max_chars].strip()
            index = example[label_col]
            if not body or not isinstance(index, int) or not 0 <= index < len(names):
                skipped += 1
                continue
            key = body.casefold()
            if key in seen:
                if seen[key] != split:
                    leaked += 1
                continue
            seen[key] = split
            row = make_row(
                dataset,
                question,
                body,
                names[index],
                names,
                rng,
                min_options,
                max_options,
                all_options,
            )
            # Never write a row the service would reject.
            EvalExample.model_validate(row)
            rows.append(row)
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        counts[split] = len(rows)

    meta = {
        "dataset": dataset,
        "revision": revision,
        "question": question,
        "classes": len(names),
        "class_names": names,
        "text_column": text_col,
        "label_column": label_col,
        "options": "all" if all_options else f"{min_options}-{max_options} sampled",
        "seed": seed,
        "skipped_rows": skipped,
        "cross_split_duplicates_dropped": leaked,
        "note": (
            "Public dataset reshaped into decision rows. A head trained here learns this "
            "dataset's distribution, which transfers to your own task only as far as the two "
            "resemble each other."
        ),
        **counts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
