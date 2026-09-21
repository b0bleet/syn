import pytest

from syn.prompt import LABELS, PromptBuilder
from syn.schema import ScoreRequest

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any cache miss means this test needs a download
        pytest.skip(f"Qwen3 tokenizer not in the local cache: {exc}")


def test_all_labels_are_single_tokens_via_fast_path(tokenizer):
    builder = PromptBuilder(tokenizer, 8192)
    assert builder.suffix_ids is not None
    assert [builder.label_cache[label] for label in LABELS] == list(range(32, 58))
    assert "<|im_end|>" in builder.reserved and "<|im_start|>" in builder.reserved
    request = ScoreRequest.model_validate(
        {
            "context": "The customer was charged twice.\nSecond line of state.",
            "question": "Which team?",
            "options": [{"id": f"o{i}", "text": f"Team {i}"} for i in range(26)],
        }
    )
    fast = builder.prepare(request)
    assert fast.label_token_ids == list(range(32, 58))
    builder.suffix_ids = None
    assert builder.prepare(request) == fast


def test_text_format_with_real_tokenizer(tokenizer):
    builder = PromptBuilder(tokenizer, 8192, "text")
    request = ScoreRequest.model_validate(
        {
            "context": "line one\nline two",
            "question": "Which?",
            "options": [{"id": "a", "text": "first\nwrapped"}, {"id": "b", "text": "second"}],
        }
    )
    prompt = builder.prepare(request)
    assert prompt.label_token_ids == [32, 33]
    assert "A. first\n   wrapped\nB. second" in tokenizer.decode(prompt.input_ids)
