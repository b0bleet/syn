"""Frozen-backbone features for the trainable head (open-jev's Route A, jevlike's recipe).

The context is rendered exactly as the pmi readout renders it, without the option listing, and
every token's final hidden state is kept. Each option is encoded on its own and mean-pooled to
one vector. Options are deliberately not encoded as continuations of the context: if they were,
the match would leak into the option vectors, the head would reach 100% in one epoch, and the
shuffled-context control would stop meaning anything. Features are cached to one .npz so head
training never touches the backbone again.
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


def extract_with(
    backend, builder: PromptBuilder, examples: list[EvalExample], out: Path, meta: dict
) -> dict:
    """Write ctx_i (Lc, H) and opt_i (N, H) float16 arrays plus labels for every example."""
    if out.exists():
        raise FileExistsError(out)
    arrays: dict[str, np.ndarray] = {}
    labels = []
    started = time.perf_counter()
    hidden = None
    for index, example in enumerate(examples):
        cloze = builder.prepare_cloze(example.request, list_options=False)
        result = backend.features(cloze.prefix_ids, cloze.span_ids)
        hidden = int(result.context.shape[1])
        arrays[f"ctx_{index}"] = result.context
        arrays[f"opt_{index}"] = result.options
        ids = [option.id for option in example.request.options]
        labels.append(ids.index(example.expected_option_id))
    meta = {
        "version": HEAD_VERSION,
        "n": len(examples),
        "hidden": hidden,
        "seconds": round(time.perf_counter() - started, 1),
        **meta,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        labels=np.array(labels, dtype=np.int32),
        meta=np.array(json.dumps(meta)),
        **arrays,
    )
    return meta


def extract_dataset(settings: Settings, dataset: Path, out: Path, limit: int = 0) -> dict:
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
    return extract_with(backend, builder, examples, out, meta)
