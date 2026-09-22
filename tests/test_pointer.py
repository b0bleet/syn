import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import CharacterTokenizer
from test_local_backend import save_tiny_qwen3

from syn.backends import LocalBackend, PointerResult, pointer_layout, pointer_mask
from syn.config import Settings
from syn.pointer import (
    PointerHead,
    PointerReadout,
    augment_options,
    delimiter_ids,
    hold_out_selection,
    source_weights,
)
from syn.prompt import PromptBuilder
from syn.schema import EvalExample, Option, ScoreRequest
from syn.scoring import Scorer
from syn.training import Augment

DELIMS = (30, 31, 29)


def _row(source: str, label: str = "yes") -> EvalExample:
    return EvalExample(
        request=ScoreRequest(
            context="c",
            question="q",
            options=[Option(id="yes", text="Yes"), Option(id="no", text="No")],
        ),
        expected_option_id=label,
        source=source,
    )


def test_source_weights_give_each_source_the_same_total():
    rows = [_row("big")] * 4 + [_row("small")]
    weights = source_weights(rows)
    assert abs(sum(weights) / len(weights) - 1) < 1e-9
    big = sum(weights[:4])
    small = weights[4]
    assert abs(big - small) < 1e-9
    assert small > weights[0]


def test_hold_out_selection_removes_one_source_from_training():
    train = [_row("a")] * 2 + [_row("b")] * 4 + [_row("c")] * 6
    validation = [_row("a"), _row("b"), _row("c")]
    kept, select, source = hold_out_selection(train, validation, seed=0)
    assert source == "a"
    assert {row.source for row in kept} == {"b", "c"}
    assert {row.source for row in select} == {"a"}


def test_pointer_layout_and_mask():
    ids, pos, seg, ends, decide = pointer_layout([1, 2], [[3], [4, 5]], DELIMS)
    assert ids == [1, 2, 30, 3, 31, 30, 4, 5, 31, 29]
    assert pos == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]  # spans share positions; decide past the longest
    assert seg == [0, 0, 1, 1, 1, 2, 2, 2, 2, -1]
    assert ends == [4, 8] and decide == 9
    allowed = pointer_mask(seg, torch.float32, "cpu")[0, 0] == 0
    assert allowed[decide].all()  # the decide token sees everything
    assert allowed[6, 3].item() is False  # span 2 cannot see span 1
    assert allowed[3, :2].all() and allowed[3, 3] and not allowed[3, 4]  # causal inside a span
    assert not allowed[1, 2:].any()  # the prefix sees no span and never the decide token
    assert not allowed[:decide, decide].any()


def test_pointer_states_are_invariant_to_option_order(tmp_path):
    save_tiny_qwen3(tmp_path / "tiny")
    settings = Settings(
        model=str(tmp_path / "tiny"),
        readout="pointer",
        pointer_path="unused",
        device="cpu",
        max_prompt_tokens=60,
    )
    backend = LocalBackend(settings)
    prefix, spans = [1, 2, 3], [[4], [5, 6], [7, 8, 9]]
    original = backend.pointer_states(prefix, spans, DELIMS)
    permuted = backend.pointer_states(prefix, [spans[2], spans[0], spans[1]], DELIMS)
    assert original.options.shape == (3, 16) and original.decide.shape == (16,)
    assert np.allclose(original.decide, permuted.decide, atol=1e-4)
    assert np.allclose(original.options[[2, 0, 1]], permuted.options, atol=1e-4)
    # Each option's state is what it would be alone with the prefix: no option sees another.
    alone = backend.pointer_states(prefix, [spans[1]], DELIMS)
    assert np.allclose(alone.options[0], original.options[1], atol=1e-4)


def test_pointer_head_round_trip_and_readout_checks(tmp_path):
    torch.manual_seed(0)
    head = PointerHead(16, 4)
    decide, options = torch.randn(2, 16), torch.randn(2, 3, 16)
    logits = head(decide, options)
    assert logits.shape == (2, 3)
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(head(decide, options[:, perm]), logits[:, perm], atol=1e-6)
    path = tmp_path / "pointer.safetensors"
    head.save(path, {"temperature": 1.7, "delimiters": {"ids": list(DELIMS)}})
    loaded, _config = PointerHead.load(path)
    assert torch.allclose(loaded(decide, options), logits)
    assert json.loads(path.with_suffix(".json").read_text())["dim"] == 4

    class Tokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|box_start|>": 30, "<|box_end|>": 31, "<|object_ref_start|>": 29}[token]

    readout = PointerReadout.load(path, Tokenizer(), 16)
    assert readout.temperature == 1.7 and readout.delimiters == DELIMS and len(readout.sha256) == 64
    with pytest.raises(ValueError, match="hidden size"):
        PointerReadout.load(path, Tokenizer(), 32)
    head.save(tmp_path / "other.safetensors", {"delimiters": {"ids": [1, 2, 3]}})
    with pytest.raises(ValueError, match="delimiter ids"):
        PointerReadout.load(tmp_path / "other.safetensors", Tokenizer(), 16)

    class Unknown:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 0

    with pytest.raises(ValueError, match="lacks the delimiter"):
        delimiter_ids(Unknown())


class PointerBackend:
    def __init__(self):
        self.calls = []

    def pointer_states(self, prefix_ids, spans, delimiters):
        self.calls.append((prefix_ids, spans, delimiters))
        rng = np.random.default_rng(0)
        return PointerResult(
            rng.normal(size=16).astype(np.float32),
            np.stack([np.random.default_rng(sum(s)).normal(size=16) for s in spans]).astype(
                np.float32
            ),
            compute_ms=1.0,
            wait_ms=0.0,
        )

    def close(self):
        pass


def test_scorer_pointer_readout(request_data, tmp_path):
    torch.manual_seed(0)
    head = PointerHead(16, 4)
    path = tmp_path / "pointer.safetensors"
    head.save(path, {"temperature": 2.0, "delimiters": {"ids": list(DELIMS)}})
    readout = PointerReadout.load(
        path,
        type(
            "T",
            (),
            {
                "unk_token_id": None,
                "convert_tokens_to_ids": staticmethod(
                    lambda t: dict(
                        zip(("<|box_start|>", "<|box_end|>", "<|object_ref_start|>"), DELIMS)
                    )[t]
                ),
            },
        )(),
        16,
    )
    settings = Settings(readout="pointer", pointer_path=path)
    builder = PromptBuilder(CharacterTokenizer(), settings.max_prompt_tokens)
    backend = PointerBackend()
    scorer = Scorer(settings, builder, backend, "abc", pointer=readout)
    response = scorer.score(ScoreRequest.model_validate(request_data))
    assert response.readout == "pointer" and response.prompt_version == "qwen-pointer-v1"
    assert response.temperature == 2.0 and response.head_sha256 == readout.sha256
    assert backend.calls[0][2] == DELIMS and len(backend.calls[0][1]) == 2
    assert abs(sum(s.probability for s in response.scores) - 1) < 1e-6
    # Reordering the options reorders the scores and nothing else.
    swapped = dict(request_data, options=list(reversed(request_data["options"])))
    again = scorer.score(ScoreRequest.model_validate(swapped))
    assert [s.log_probability for s in again.scores] == [
        s.log_probability for s in reversed(response.scores)
    ]
    with pytest.raises(ValueError, match="needs a loaded PointerReadout"):
        Scorer(settings, builder, backend, "abc")


def test_augment_options_at_the_text_level():
    options = [Option(id=str(i), text=f"option {i}") for i in range(4)]
    import random

    kept, label, changed = augment_options(options, 1, Augment(p_none=1.0), random.Random(0))
    assert changed and label == 3 and kept[3].id == "__none__"
    assert [o.id for o in kept[:3]] == ["0", "2", "3"]
    two = options[:2]
    assert augment_options(two, 1, Augment(p_none=1.0), random.Random(0)) == (two, 1, False)
    more, label, _ = augment_options(options, 2, Augment(p_none_distract=1.0), random.Random(0))
    assert label == 2 and len(more) == 5 and more[4].id == "__none__"
    more, label, _ = augment_options(options, 2, Augment(p_distract=1.0), random.Random(0))
    assert label == 2 and more[4].id == "__distractor__"
    assert augment_options(options, 2, Augment(), random.Random(0)) == (options, 2, False)


def _tokenizer_cached() -> bool:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return (home / "hub" / "models--Qwen--Qwen3-0.6B").is_dir()


@pytest.mark.skipif(not _tokenizer_cached(), reason="Qwen3 tokenizer is not cached")
def test_train_pointer_end_to_end_on_a_tiny_backbone(tmp_path):
    pytest.importorskip("peft")
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from syn.pointer import train_pointer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
    )
    base = tmp_path / "base"
    Qwen3ForCausalLM(config).save_pretrained(base)
    tokenizer.save_pretrained(base)
    rows = []
    for i in range(12):
        team = "Billing" if i % 2 else "Sales"
        rows.append(
            {
                "request": {
                    "context": f"Ticket {i}: {'refund' if i % 2 else 'pricing'} question",
                    "question": "Which team?",
                    "options": [
                        {"id": "sales", "text": "Sales"},
                        {"id": "billing", "text": "Billing"},
                        {"id": "it", "text": "IT"},
                    ],
                },
                "expected_option_id": team.lower(),
                "source": "tickets",
            }
        )
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "run"
    settings = Settings(model=str(base), device="cpu", max_prompt_tokens=400)
    result = train_pointer(
        data,
        data,
        out,
        rank=2,
        head_dim=4,
        epochs=1,
        batch_size=4,
        accumulate=1,
        augment=Augment(p_none=0.2, p_distract=0.2),
        ordinal_weight=0.5,
        settings=settings,
        log=lambda _: None,
    )
    assert (out / "backbone" / "config.json").exists() and (out / "pointer.json").exists()
    sidecar = json.loads((out / "pointer.json").read_text())
    assert sidecar["temperature"] == result["temperature"] > 0
    assert sidecar["delimiters"]["ids"] == list(delimiter_ids(tokenizer))
    assert sidecar["sources"] == ["tickets"] and sidecar["history"][0]["epoch"] == 1

    # The saved run serves through the ordinary local backend with the pointer readout.
    served = Settings(
        model=str(out / "backbone"),
        readout="pointer",
        pointer_path=out / "pointer.safetensors",
        device="cpu",
        max_prompt_tokens=400,
    )
    from syn.backends import resolve_config

    backend = LocalBackend(served, resolve_config(served))
    builder = PromptBuilder(tokenizer, served.max_prompt_tokens)
    readout = PointerReadout.load(served.pointer_path, tokenizer, 16)
    scorer = Scorer(served, builder, backend, None, pointer=readout)
    response = scorer.score(ScoreRequest.model_validate(rows[0]["request"]))
    assert response.readout == "pointer" and response.temperature == sidecar["temperature"]
    assert len(response.scores) == 3
    with pytest.raises(FileExistsError):
        train_pointer(data, data, out, settings=settings, log=lambda _: None)
