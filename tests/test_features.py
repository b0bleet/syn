from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import CharacterTokenizer

from syn.backends import FeatureResult
from syn.features import (
    DISTRACTOR_TEXTS,
    NONE_TEXTS,
    default_source,
    extract_with,
)
from syn.prompt import PromptBuilder
from syn.schema import EvalExample
from syn.training import FeatureSet

HIDDEN = 6


class FeatureBackend:
    """Deterministic: every option vector is a function of its own tokens, nothing else."""

    def features(self, prefix_ids, spans):
        ctx = np.random.default_rng(len(prefix_ids)).normal(size=(len(prefix_ids), HIDDEN))
        opts = [np.random.default_rng(sum(span)).normal(size=HIDDEN) for span in spans]
        return FeatureResult(
            ctx.astype(np.float16), np.stack(opts).astype(np.float16), compute_ms=1.0, wait_ms=0.0
        )

    def close(self):
        pass


def test_extract_with_writes_tags_and_extra_vectors(tmp_path):
    builder = PromptBuilder(CharacterTokenizer(), 4096)
    backend = FeatureBackend()
    examples = [
        EvalExample.model_validate(
            {
                "request": {
                    "context": "poor",
                    "question": "Rate it",
                    "options": [{"id": "0", "text": "bad"}, {"id": "1", "text": "good"}],
                },
                "expected_option_id": "0",
                "ordinal": True,
                "source": "reviews",
            }
        ),
        EvalExample.model_validate(
            {
                "request": {
                    "context": "Refund please",
                    "question": "Team?",
                    "options": [{"id": "s", "text": "Sales"}, {"id": "b", "text": "Billing"}],
                },
                "expected_option_id": "b",
            }
        ),
    ]
    out = tmp_path / "f.npz"
    meta = extract_with(backend, builder, examples, out, {"model": "toy"}, source="tickets")
    assert meta["sources"] == ["reviews", "tickets"] and meta["ordinal_rows"] == 1
    assert meta["none_texts"] == list(NONE_TEXTS) and meta["hidden"] == HIDDEN
    archive = np.load(out, allow_pickle=False)
    assert archive["labels"].tolist() == [0, 1]
    assert archive["ordinal"].tolist() == [True, False]
    assert archive["sources"].tolist() == ["reviews", "tickets"]
    assert archive["none"].shape == (len(NONE_TEXTS), HIDDEN)
    assert archive["distractors"].shape == (len(DISTRACTOR_TEXTS), HIDDEN)
    # Encoded like any option: the same text served as an option gives the same vector.
    served = backend.features([1], [builder.option_ids(NONE_TEXTS[0])]).options[0]
    assert np.array_equal(archive["none"][0], served)
    loaded = FeatureSet(out)
    assert len(loaded) == 2 and loaded.sources_present == ["reviews", "tickets"]
    assert loaded.none.shape[0] == len(NONE_TEXTS) and loaded.distractors.shape[0] == len(
        DISTRACTOR_TEXTS
    )
    with pytest.raises(FileExistsError):
        extract_with(backend, builder, examples, out, {})
    with pytest.raises(ValueError, match="No examples"):
        extract_with(backend, builder, [], tmp_path / "empty.npz", {})


def test_default_source_is_the_directory_or_the_file():
    assert default_source(Path("data/agnews/train.jsonl")) == "agnews"
    assert default_source(Path("data/spam-eval.jsonl")) == "spam-eval"
    assert default_source(Path("data/eval/agnews-250.jsonl")) == "agnews-250"
    assert default_source(Path("rows.jsonl")) == "rows"
