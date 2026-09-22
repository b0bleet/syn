from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SYN_", allow_inf_nan=False)
    backend: Literal["local", "sglang"] = "local"
    # A Hub model id, a local directory, or hf://<user>/<repo>/<directory> (a pointer backbone).
    model: str = "Qwen/Qwen3-0.6B"
    revision: str = "main"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    # auto = float32 on CPU, bfloat16 elsewhere. Qwen3 is trained in bfloat16; float16 can overflow.
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    prompt_format: Literal["json", "text"] = "json"
    # letters: read the option letter's next-token log-prob. One ordering by default.
    # pmi: score each option's own text likelihood under a block-causal mask (all options in one
    # forward), minus the same likelihood with the context removed. Local backend only.
    # head: a trained AttentionHead over frozen backbone features (see `syn train-head`).
    # Local backend only; requires head_path.
    # pointer: a backbone adapted by `syn train-pointer` (point `model` at its backbone/
    # directory) read by its pointer head. Local backend only; requires pointer_path.
    readout: Literal["letters", "pmi", "head", "pointer"] = "letters"
    # A local checkpoint, or hf://<user>/<repo>/<path>.safetensors fetched from the Hub at
    # startup together with its JSON sidecar (HF_TOKEN for a private repo).
    head_path: str | None = None
    pointer_path: str | None = None
    # pmi only. False: each option is scored from the context, the question, and its own text, so
    # the result is exactly invariant to option order by construction. True: the options are also
    # named in the prefix so the model knows the choice set, but that listing has an order and it
    # leaks; measured on the smoke set it changed 2 of 3 answers under shuffling.
    pmi_list_options: bool = False
    max_prompt_tokens: int = Field(default=8192, ge=1, le=32767)
    # Number of cyclic option orderings scored per request. 1 = the caller's order only.
    # 0 = one per option, so every option takes every position once.
    orderings: int = Field(default=1, ge=0, le=26)
    temperature: float = Field(default=1.0, gt=0, le=100)
    abstain_threshold: float = Field(default=0.0, ge=0, le=1)
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    min_ordering_agreement: float = Field(default=0.0, ge=0, le=1)
    sglang_url: str = "http://127.0.0.1:30000"
    sglang_api_key: str | None = None
    # Refuse to start if the SGLang server reports a different model than configured here.
    sglang_check_model: bool = True
    timeout_seconds: float = Field(default=120, gt=0)
    # Bearer token callers must present on /v1/score and the GET shorthand. /health stays open.
    # Unset means no authentication, which is only acceptable behind a private network.
    api_key: str | None = None
    # Token budget for one batched local forward: orderings are batched together until
    # rows x padded width would exceed this, then split, so long contexts cannot exhaust memory.
    local_batch_tokens: int = Field(default=16384, ge=1)
    log_path: Path | None = None

    @field_validator("head_path", "pointer_path", mode="before")
    @classmethod
    def checkpoint_as_text(cls, value):
        return str(value) if isinstance(value, Path) else value

    @model_validator(mode="after")
    def head_needs_a_checkpoint(self):
        if self.readout == "head" and self.head_path is None:
            raise ValueError("SYN_HEAD_PATH is required when SYN_READOUT=head")
        if self.readout == "pointer" and self.pointer_path is None:
            raise ValueError("SYN_POINTER_PATH is required when SYN_READOUT=pointer")
        return self
