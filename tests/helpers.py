import math
from typing import ClassVar

from syn.backends import BackendResult
from syn.config import Settings
from syn.prompt import PromptBuilder
from syn.scoring import Scorer


class CharacterTokenizer:
    """One token per character. The template tail starts with a control string, like Qwen's."""

    all_special_tokens: ClassVar[list[str]] = ["<|im_end|>", "<|im_start|>"]

    def __init__(self):
        self.encode_calls = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return (
            "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
            "<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def encode(self, text, **kwargs):
        self.encode_calls += 1
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FixedBackend:
    """Position-biased: the same row for every prompt, whatever the options say."""

    def __init__(self, row=None):
        self.row = list(row) if row is not None else [math.log(0.1), math.log(0.3)]
        self.calls = []

    def score(self, prompts):
        self.calls.append(prompts)
        return BackendResult(
            [self.row[: len(p.labels)] for p in prompts], compute_ms=1.0, wait_ms=0.5
        )

    def close(self):
        pass


class ContentBackend:
    """Scores each option by its text wherever it appears, so every ordering agrees."""

    def __init__(self, tokenizer, by_text):
        self.tokenizer, self.by_text = tokenizer, by_text

    def score(self, prompts):
        rows = []
        for prompt in prompts:
            text = self.tokenizer.decode(prompt.input_ids)
            ordered = sorted(self.by_text, key=text.index)
            rows.append([self.by_text[t] for t in ordered])
        return BackendResult(rows, compute_ms=1.0, wait_ms=0.0)

    def close(self):
        pass


def make_scorer(
    settings=None,
    backend=None,
    tokenizer=None,
    revision_commit="abc123",
    head=None,
    head_sha256=None,
):
    settings = settings or Settings()
    tokenizer = tokenizer or CharacterTokenizer()
    builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
    return Scorer(settings, builder, backend or FixedBackend(), revision_commit, head, head_sha256)


def billing_backend(tokenizer):
    return ContentBackend(tokenizer, {"Sales": math.log(0.1), "Billing": math.log(0.3)})
