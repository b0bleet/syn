"""Frozen-backbone features for the trainable head.

The context is rendered exactly as the pmi readout renders it, without the option listing, and
every token's final hidden state is kept. Each option is encoded on its own and mean-pooled to
one vector. Options are deliberately not encoded as continuations of the context: if they were,
the match would leak into the option vectors, the head would reach 100% in one epoch, and the
shuffled-context control would stop meaning anything. Features are cached to one .npz so head
training never touches the backbone again.

Every row also carries its source tag and ordinal flag. The file also holds vectors for a few
wordings of "None of the above" and for a few unrelated distractor texts, all encoded like any
other option, so training can add them to rows and teach the head that sometimes nothing
offered fits, and that an unrelated extra option changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .backends import LocalBackend, resolve_config, revision_commit
from .config import Settings
from .evaluation import read_jsonl
from .prompt import HEAD_VERSION, PromptBuilder
from .schema import EvalExample

# Several wordings, so a served "Other" or "Not listed" reads as none too, not only the exact text.
NONE_TEXTS = (
    "None of the above",
    "None of these",
    "Something else",
    "Not listed here",
    "No option fits",
    "Other",
)
# Unrelated to any state; added as wrong options so an extra irrelevant choice is never picked.
DISTRACTOR_TEXTS = (
    "The weather on Tuesday",
    "A shade of green",
    "Instructions for baking bread",
    "Unrelated: a parking permit renewal",
)
# Directory names that hold a collection rather than one source.
GENERIC_DIRECTORIES = {"", ".", "data", "eval"}


def default_source(dataset: Path) -> str:
    """data/agnews/train.jsonl -> agnews; data/spam-eval.jsonl -> spam-eval."""
    parent = dataset.parent.name
    return dataset.stem if parent in GENERIC_DIRECTORIES else parent


def extract_with(
    backend,
    builder: PromptBuilder,
    examples: list[EvalExample],
    out: Path,
    meta: dict,
    source: str | None = None,
) -> dict:
    """Write ctx_i (Lc, H) and opt_i (N, H) float16 arrays plus labels for every example."""
    if out.exists():
        raise FileExistsError(out)
    if not examples:
        raise ValueError("No examples to extract features for")
    arrays: dict[str, np.ndarray] = {}
    labels, ordinal, sources = [], [], []
    started = time.perf_counter()
    hidden = None
    extra = None
    for index, example in enumerate(examples):
        cloze = builder.prepare_cloze(example.request, list_options=False)
        result = backend.features(cloze.prefix_ids, cloze.span_ids)
        hidden = int(result.context.shape[1])
        arrays[f"ctx_{index}"] = result.context
        arrays[f"opt_{index}"] = result.options
        ids = [option.id for option in example.request.options]
        labels.append(ids.index(example.expected_option_id))
        ordinal.append(example.ordinal)
        sources.append(example.source or source or "")
        if extra is None:
            # Encoded standalone like every option, so a served "None of the above" matches.
            spans = [builder.option_ids(text) for text in NONE_TEXTS + DISTRACTOR_TEXTS]
            extra = np.asarray(backend.features(cloze.prefix_ids, spans).options, dtype=np.float16)
    meta = {
        "version": HEAD_VERSION,
        "n": len(examples),
        "hidden": hidden,
        "seconds": round(time.perf_counter() - started, 1),
        "source": source,
        "sources": sorted(set(sources)),
        "ordinal_rows": sum(ordinal),
        "none_texts": list(NONE_TEXTS),
        "distractor_texts": list(DISTRACTOR_TEXTS),
        **meta,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        labels=np.array(labels, dtype=np.int32),
        ordinal=np.array(ordinal, dtype=bool),
        sources=np.array(sources, dtype=str),
        none=extra[: len(NONE_TEXTS)],
        distractors=extra[len(NONE_TEXTS) :],
        meta=np.array(json.dumps(meta)),
        **arrays,
    )
    return meta


def extract_dataset(
    settings: Settings, dataset: Path, out: Path, limit: int = 0, source: str | None = None
) -> dict:
    """Load the configured Qwen3 locally and cache features for a labeled JSONL dataset."""
    from transformers import AutoTokenizer

    if settings.backend != "local":
        raise ValueError("Feature extraction needs hidden states; use the local backend")
    examples = [EvalExample.model_validate(row) for row in read_jsonl(dataset)]
    if limit:
        examples = examples[:limit]
    config = resolve_config(settings)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model, revision=settings.revision, trust_remote_code=False
    )
    builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
    backend = LocalBackend(settings, config)
    meta = {
        "model": settings.model,
        "revision": settings.revision,
        "revision_commit": revision_commit(config),
        "dtype": str(backend.dtype).replace("torch.", ""),
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    return extract_with(backend, builder, examples, out, meta, source or default_source(dataset))
