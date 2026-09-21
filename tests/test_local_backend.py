import pytest

from syn.backends import (
    LocalBackend,
    block_mask,
    isolation_layout,
    resolve_config,
    revision_commit,
)
from syn.config import Settings
from syn.prompt import PreparedPrompt

torch = pytest.importorskip("torch")


def save_tiny_qwen3(path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = Qwen3ForCausalLM(config).eval()
    model.save_pretrained(path)
    return model


def test_local_backend_loads_and_matches_full_forward(tmp_path):
    reference = save_tiny_qwen3(tmp_path)
    settings = Settings(model=str(tmp_path), device="cpu", max_prompt_tokens=32)
    config = resolve_config(settings)
    assert revision_commit(config) is None  # a local directory has no hub commit
    backend = LocalBackend(settings, config)
    assert backend.dtype == torch.float32
    prompts = [
        PreparedPrompt([1, 2, 3, 4], ["A", "B"], [10, 11]),
        PreparedPrompt([5, 6, 7], ["A", "B", "C"], [10, 11, 12]),
    ]
    result = backend.score(prompts)
    assert result.wait_ms is not None and result.wait_ms >= 0 and result.compute_ms > 0
    with torch.inference_mode():
        for prompt, row in zip(prompts, result.log_probs, strict=True):
            logits = reference(input_ids=torch.tensor([prompt.input_ids])).logits[0, -1]
            expected = logits.log_softmax(-1)[prompt.label_token_ids].tolist()
            assert row == pytest.approx(expected, abs=1e-5)


def test_local_backend_rejects_other_models_before_loading_weights(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    Qwen2ForCausalLM(config).save_pretrained(tmp_path)
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: pytest.fail("weights must not load for an unsupported model"),
    )
    with pytest.raises(ValueError, match="Qwen3"):
        LocalBackend(Settings(model=str(tmp_path), device="cpu"))


def test_isolation_layout_and_mask():
    ids, pos, seg, starts = isolation_layout([1, 2, 3], [[4, 5], [6], [7, 8, 9]])
    assert ids == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert pos == [0, 1, 2, 3, 4, 3, 3, 4, 5]  # every span restarts at the prefix length
    assert seg == [0, 0, 0, 1, 1, 2, 3, 3, 3]
    assert starts == [3, 5, 6]
    mask = block_mask(seg, torch.float32, "cpu")[0, 0]
    allowed = mask == 0
    assert allowed[4, 3] and allowed[4, 0]  # span 1 sees itself and the prefix
    assert not allowed[4, 5] and not allowed[8, 4]  # spans never see each other
    assert not allowed[2, 3] and allowed[8, 6] and allowed[8, 8]
    assert bool(allowed.diagonal().all())


def test_span_scores_match_independent_forwards(tmp_path):
    """The block mask must reproduce, in one forward, what separate prefix+span forwards give."""
    reference = save_tiny_qwen3(tmp_path)
    settings = Settings(model=str(tmp_path), device="cpu", max_prompt_tokens=32, readout="pmi")
    backend = LocalBackend(settings)
    prefix, spans = [1, 2, 3, 4, 5], [[6, 7], [8], [9, 10, 11]]
    result = backend.score_spans(prefix, spans)
    assert len(result.log_probs) == 1 and len(result.log_probs[0]) == 3
    assert result.wait_ms is not None and result.compute_ms > 0
    with torch.inference_mode():
        for span, got in zip(spans, result.log_probs[0], strict=True):
            log_probs = reference(input_ids=torch.tensor([prefix + span])).logits[0].log_softmax(-1)
            expected = sum(
                log_probs[len(prefix) - 1 + j, span[j]].item() for j in range(len(span))
            ) / len(span)
            assert got == pytest.approx(expected, abs=1e-4)


def test_features_match_standalone_forwards(tmp_path):
    reference = save_tiny_qwen3(tmp_path)
    backend = LocalBackend(Settings(model=str(tmp_path), device="cpu", max_prompt_tokens=32))
    prefix, spans = [1, 2, 3, 4], [[5, 6], [7], [8, 9, 10]]
    result = backend.features(prefix, spans)
    assert result.context.shape == (4, 16) and result.context.dtype.name == "float16"
    assert result.options.shape == (3, 16) and result.wait_ms is not None
    with torch.inference_mode():
        expected_ctx = reference.model(input_ids=torch.tensor([prefix])).last_hidden_state[0]
        assert torch.allclose(
            torch.from_numpy(result.context.astype("float32")), expected_ctx, atol=2e-2
        )
        for span, pooled in zip(spans, result.options, strict=True):
            hidden = reference.model(input_ids=torch.tensor([span])).last_hidden_state[0]
            assert torch.allclose(
                torch.from_numpy(pooled.astype("float32")), hidden.mean(0), atol=2e-2
            )


def test_batched_letters_match_individual_forwards_with_and_without_chunking(tmp_path):
    """Left-padded batching must reproduce per-prompt forwards for prompts of different lengths."""
    reference = save_tiny_qwen3(tmp_path)
    prompts = [
        PreparedPrompt([1, 2, 3, 4], ["A", "B"], [10, 11]),
        PreparedPrompt([5, 6, 7], ["A", "B", "C"], [10, 11, 12]),
        PreparedPrompt([8, 9, 10, 11, 12, 13], ["A", "B"], [14, 15]),
    ]
    with torch.inference_mode():
        expected = [
            reference(input_ids=torch.tensor([p.input_ids]))
            .logits[0, -1]
            .log_softmax(-1)[p.label_token_ids]
            .tolist()
            for p in prompts
        ]
    # A budget of 8 tokens forces chunks [3-wide, 4-wide] and [6-wide]; 16384 keeps one batch.
    for budget, expected_chunks in ((8, 2), (16384, 1)):
        settings = Settings(
            model=str(tmp_path), device="cpu", max_prompt_tokens=32, local_batch_tokens=budget
        )
        backend = LocalBackend(settings)
        assert len(backend._chunks(prompts)) == expected_chunks
        result = backend.score(prompts)
        assert len(result.log_probs) == 3
        for row, want in zip(result.log_probs, expected, strict=True):
            assert row == pytest.approx(want, abs=1e-4)


def test_local_backend_context_window_and_dtype(tmp_path):
    save_tiny_qwen3(tmp_path)
    with pytest.raises(ValueError, match="context window"):
        LocalBackend(Settings(model=str(tmp_path), device="cpu", max_prompt_tokens=64))
    backend = LocalBackend(
        Settings(model=str(tmp_path), device="cpu", dtype="bfloat16", max_prompt_tokens=32)
    )
    assert next(backend.model.parameters()).dtype == torch.bfloat16
