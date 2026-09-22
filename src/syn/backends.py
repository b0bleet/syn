import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings
from .prompt import PreparedPrompt

# SGLang reports exp(logprob) and sends 0.0 for -inf. Below this the probability underflows
# float64, so such labels are floored rather than treated as a backend failure.
LOG_PROB_FLOOR = -745.0


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendResult:
    # One row per prompt, one full-vocabulary log probability per label token.
    log_probs: list[list[float]]
    compute_ms: float
    # Time spent waiting for the local model lock. None for remote backends, whose queueing
    # happens inside the server.
    wait_ms: float | None = None


@dataclass(frozen=True)
class FeatureResult:
    # Final hidden state of every context token, (Lc, H) float16.
    context: object
    # Mean-pooled hidden state of each option encoded on its own, (N, H) float16.
    options: object
    compute_ms: float
    wait_ms: float | None = None


@dataclass(frozen=True)
class PointerResult:
    # Final hidden state of the decide token, (H,) float32.
    decide: object
    # Final hidden state of each option's closing delimiter, (N, H) float32.
    options: object
    compute_ms: float
    wait_ms: float | None = None


class Backend(Protocol):
    def score(self, prompts: list[PreparedPrompt]) -> BackendResult: ...

    def score_spans(self, prefix_ids: list[int], spans: list[list[int]]) -> BackendResult:
        """One row: the mean token log probability of each span given the prefix alone."""
        ...

    def features(self, prefix_ids: list[int], spans: list[list[int]]) -> FeatureResult:
        """Frozen-backbone features for the trainable head."""
        ...

    def pointer_states(
        self, prefix_ids: list[int], spans: list[list[int]], delimiters: tuple[int, int, int]
    ) -> PointerResult:
        """Decide and option states for the pointer readout."""
        ...

    def close(self) -> None: ...


def isolation_layout(prefix: list[int], spans: list[list[int]]):
    """Pack prefix and spans into one sequence where every span starts at the same position.

    Returns ids, position ids, segment ids (0 = prefix, k = span k), and each span's start index.
    With the matching mask, each span's tokens see only the prefix and their own span, so their
    likelihoods equal what separate prefix+span forwards would give (option isolation).
    """
    p = len(prefix)
    ids, pos, seg, starts = list(prefix), list(range(p)), [0] * p, []
    for k, span in enumerate(spans, 1):
        starts.append(len(ids))
        ids += span
        pos += range(p, p + len(span))
        seg += [k] * len(span)
    return ids, pos, seg, starts


def block_mask(seg: list[int], dtype, device):
    """Additive [1, 1, L, L] mask: attend(i, j) iff j <= i and (key is prefix or same segment)."""
    import torch

    s = torch.tensor(seg, device=device)
    length = len(seg)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same = (s[None, :] == s[:, None]) | (s[None, :] == 0)
    allow = (causal & same) | torch.eye(length, dtype=torch.bool, device=device)
    mask = torch.zeros(length, length, dtype=dtype, device=device)
    return mask.masked_fill(~allow, torch.finfo(dtype).min)[None, None]


DECIDE_SEGMENT = -1


def pointer_layout(prefix: list[int], spans: list[list[int]], delimiters: tuple[int, int, int]):
    """The pointer readout's sequence: prefix, every option wrapped in open and close delimiters
    as its own isolated span, then one decide token.

    Returns ids, position ids, segment ids (0 = prefix, k = span k, -1 = decide), the index of
    each span's closing delimiter, and the decide token's index. Spans share positions, so the
    decide token sits one past the longest span and sees the same thing whatever the order.
    """
    opened, closed, decide = delimiters
    wrapped = [[opened, *span, closed] for span in spans]
    ids, pos, seg, starts = isolation_layout(prefix, wrapped)
    ends = [start + len(span) - 1 for start, span in zip(starts, wrapped, strict=True)]
    ids.append(decide)
    pos.append(len(prefix) + max(len(span) for span in wrapped))
    seg.append(DECIDE_SEGMENT)
    return ids, pos, seg, ends, len(ids) - 1


def pointer_mask(seg: list[int], dtype, device):
    """block_mask, except that the decide token (segment -1) attends to every earlier token."""
    import torch

    s = torch.tensor(seg, device=device)
    length = len(seg)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same = (s[None, :] == s[:, None]) | (s[None, :] == 0) | (s[:, None] == DECIDE_SEGMENT)
    allow = (causal & same) | torch.eye(length, dtype=torch.bool, device=device)
    mask = torch.zeros(length, length, dtype=dtype, device=device)
    return mask.masked_fill(~allow, torch.finfo(dtype).min)[None, None]


def causal_text_config(config):
    """The text model Qwen3.5 actually scores with.

    The Hub checkpoint is a multimodal wrapper (`qwen3_5`). Letter logits come from its text
    config (`qwen3_5_text`). The commit pin lives on the wrapper, so it is copied across.
    """
    commit = revision_commit(config)
    if getattr(config, "model_type", None) == "qwen3_5" and hasattr(config, "get_text_config"):
        text = config.get_text_config()
        if commit and not revision_commit(text):
            text._commit_hash = commit
        return text
    return config


def require_local_readout(model_type: str, readout: str) -> None:
    """Qwen3.5 is letters-only. Its linear-attention layers ignore the 4D option mask."""
    if model_type == "qwen3_5_text":
        if readout != "letters":
            raise ValueError(
                "Qwen3.5 serves the letters readout only. "
                "The pmi, head, and pointer readouts stay on Qwen3."
            )
        return
    if model_type != "qwen3":
        raise ValueError(
            f"The local backend supports Qwen3 and Qwen3.5 text models, not {model_type}"
        )


def resolve_config(settings: Settings):
    """Fetch only the model config, so the checkpoint can be checked and pinned before weights load."""
    from transformers import AutoConfig

    from .artifacts import pretrained_call

    model, extra = pretrained_call(settings.model, settings.revision)
    config = AutoConfig.from_pretrained(model, trust_remote_code=False, **extra)
    return causal_text_config(config)


def _load_causal_lm(model: str, config, dtype, attn, extra):
    from transformers import AutoModelForCausalLM

    kwargs = dict(
        config=config,
        dtype=dtype,
        trust_remote_code=False,
        attn_implementation=attn,
        **extra,
    )
    if getattr(config, "model_type", None) != "qwen3_5_text":
        return AutoModelForCausalLM.from_pretrained(model, **kwargs)
    try:
        from transformers import Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("Qwen3.5 needs transformers>=5.17") from exc
    loaded, info = Qwen3_5ForCausalLM.from_pretrained(
        model, output_loading_info=True, **kwargs
    )
    # Vision weights in the multimodal checkpoint are not part of the text model.
    dropped = {
        key: info.get(key)
        for key in ("missing_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if dropped:
        raise RuntimeError(f"Qwen3.5 text weights did not load completely: {dropped}")
    return loaded


def revision_commit(config) -> str | None:
    commit = getattr(config, "_commit_hash", None)
    return commit if isinstance(commit, str) and commit else None


class LocalBackend:
    def __init__(self, settings: Settings, config=None):
        import torch

        from .artifacts import pretrained_call

        config = causal_text_config(config if config is not None else resolve_config(settings))
        require_local_readout(config.model_type, settings.readout)
        if settings.max_prompt_tokens >= config.max_position_embeddings:
            raise ValueError("max_prompt_tokens must leave room within the model context window")
        device = settings.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        dtype_name = settings.dtype
        if dtype_name == "auto":
            dtype_name = "float32" if device == "cpu" else "bfloat16"
        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype_name)
        self.batch_tokens = settings.local_batch_tokens
        self.lock = threading.Lock()
        # The pmi readout passes an explicit 4D mask. SDPA accepts it on CUDA; eager is the
        # known-good path elsewhere. Other readouts keep the library default.
        attn = None
        if settings.readout in ("pmi", "pointer"):
            attn = "sdpa" if device == "cuda" else "eager"
        model, extra = pretrained_call(settings.model, settings.revision)
        self.model = _load_causal_lm(model, config, self.dtype, attn, extra).to(device).eval()

    def _chunks(self, prompts: list[PreparedPrompt]) -> list[list[int]]:
        """Group prompt indices so that rows x padded width stays within the token budget.

        Sorted by length, so the last prompt added to a chunk is its widest and the padding
        waste is small.
        """
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i].input_ids))
        chunks: list[list[int]] = []
        current: list[int] = []
        for index in order:
            width = len(prompts[index].input_ids)
            if current and width * (len(current) + 1) > self.batch_tokens:
                chunks.append(current)
                current = []
            current.append(index)
        if current:
            chunks.append(current)
        return chunks

    def score(self, prompts: list[PreparedPrompt]) -> BackendResult:
        """All orderings in as few forwards as the token budget allows.

        Prompts are left-padded so every prompt's final token sits at the last position, which is
        the one position whose logits we read. Explicit position ids keep each real token at the
        position it would have had unpadded, so the result equals the unbatched computation up to
        floating-point batch effects. One model per process; forwards are serialized.
        """
        torch = self.torch
        queued = time.perf_counter()
        rows: list[list[float] | None] = [None] * len(prompts)
        chunks = self._chunks(prompts)
        with self.lock:
            acquired = time.perf_counter()
            with torch.inference_mode():
                for chunk in chunks:
                    width = max(len(prompts[i].input_ids) for i in chunk)
                    ids = torch.zeros((len(chunk), width), dtype=torch.long, device=self.device)
                    mask = torch.zeros_like(ids)
                    for row, index in enumerate(chunk):
                        sequence = prompts[index].input_ids
                        ids[row, width - len(sequence) :] = torch.tensor(
                            sequence, dtype=torch.long, device=self.device
                        )
                        mask[row, width - len(sequence) :] = 1
                    positions = (mask.cumsum(-1) - 1).clamp(min=0)
                    output = self.model(
                        input_ids=ids,
                        attention_mask=mask,
                        position_ids=positions,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    log_probs = output.logits[:, -1].float().log_softmax(dim=-1)
                    for row, index in enumerate(chunk):
                        rows[index] = log_probs[row, prompts[index].label_token_ids].cpu().tolist()
            finished = time.perf_counter()
        rows = [row for row in rows if row is not None]
        return BackendResult(
            rows,
            compute_ms=(finished - acquired) * 1000,
            wait_ms=(acquired - queued) * 1000,
        )

    def score_spans(self, prefix_ids: list[int], spans: list[list[int]]) -> BackendResult:
        torch = self.torch
        queued = time.perf_counter()
        ids, pos, seg, starts = isolation_layout(prefix_ids, spans)
        prefix_end = len(prefix_ids) - 1
        # Positions whose logits we read: the prefix end predicts every span's first token; each
        # span's own tokens predict the rest. Keeping only these avoids materializing L x vocab.
        keep = sorted(
            {prefix_end, *(s + j for s, span in zip(starts, spans) for j in range(len(span) - 1))}
        )
        index = {position: i for i, position in enumerate(keep)}
        with self.lock:
            acquired = time.perf_counter()
            with torch.inference_mode():
                output = self.model(
                    input_ids=torch.tensor([ids], dtype=torch.long, device=self.device),
                    position_ids=torch.tensor([pos], dtype=torch.long, device=self.device),
                    attention_mask=block_mask(seg, self.dtype, self.device),
                    use_cache=False,
                    logits_to_keep=torch.tensor(keep, dtype=torch.long, device=self.device),
                )
                log_probs = output.logits[0].float().log_softmax(dim=-1).cpu()
            finished = time.perf_counter()
        scores = []
        for start, span in zip(starts, spans, strict=True):
            total = log_probs[index[prefix_end], span[0]].item()
            for j in range(1, len(span)):
                total += log_probs[index[start + j - 1], span[j]].item()
            scores.append(total / len(span))
        return BackendResult(
            [scores],
            compute_ms=(finished - acquired) * 1000,
            wait_ms=(acquired - queued) * 1000,
        )

    def features(self, prefix_ids: list[int], spans: list[list[int]]) -> FeatureResult:
        """Context token states from one forward; option vectors from one padded standalone batch.

        Padded option positions carry whatever the model produces, but with right padding and the
        attention mask no real token attends to them and the pooling excludes them, so the pad id
        never matters.
        """
        import numpy as np

        torch = self.torch
        queued = time.perf_counter()
        base = self.model.model  # the decoder without its vocabulary head
        width = max(len(span) for span in spans)
        with self.lock:
            acquired = time.perf_counter()
            with torch.inference_mode():
                prefix = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
                context = base(
                    input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=False
                ).last_hidden_state[0]
                ids = torch.tensor(
                    [span + [0] * (width - len(span)) for span in spans],
                    dtype=torch.long,
                    device=self.device,
                )
                mask = torch.tensor(
                    [[1] * len(span) + [0] * (width - len(span)) for span in spans],
                    dtype=torch.long,
                    device=self.device,
                )
                hidden = base(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                weights = mask[..., None].to(torch.float32)
                pooled = (hidden.float() * weights).sum(1) / weights.sum(1)
                context_np = context.float().cpu().numpy().astype(np.float16)
                options_np = pooled.cpu().numpy().astype(np.float16)
            finished = time.perf_counter()
        return FeatureResult(
            context_np,
            options_np,
            compute_ms=(finished - acquired) * 1000,
            wait_ms=(acquired - queued) * 1000,
        )

    def pointer_states(
        self, prefix_ids: list[int], spans: list[list[int]], delimiters: tuple[int, int, int]
    ) -> PointerResult:
        """One forward over the pointer layout; the decide and closing-delimiter states."""
        torch = self.torch
        queued = time.perf_counter()
        ids, pos, seg, ends, decide = pointer_layout(prefix_ids, spans, delimiters)
        base = self.model.model
        with self.lock:
            acquired = time.perf_counter()
            with torch.inference_mode():
                hidden = (
                    base(
                        input_ids=torch.tensor([ids], dtype=torch.long, device=self.device),
                        position_ids=torch.tensor([pos], dtype=torch.long, device=self.device),
                        attention_mask=pointer_mask(seg, self.dtype, self.device),
                        use_cache=False,
                    )
                    .last_hidden_state[0]
                    .float()
                )
                decide_np = hidden[decide].cpu().numpy()
                options_np = hidden[ends].cpu().numpy()
            finished = time.perf_counter()
        return PointerResult(
            decide_np,
            options_np,
            compute_ms=(finished - acquired) * 1000,
            wait_ms=(acquired - queued) * 1000,
        )

    def close(self):
        pass


class SGLangBackend:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.model
        self.client = client or httpx.Client(
            base_url=settings.sglang_url.rstrip("/") + "/",
            timeout=settings.timeout_seconds,
            headers={"Authorization": f"Bearer {settings.sglang_api_key}"}
            if settings.sglang_api_key
            else {},
        )

    def score(self, prompts: list[PreparedPrompt]) -> BackendResult:
        if not prompts:
            raise BackendError("No prompts to score")
        started = time.perf_counter()
        rows: list[list[float] | None] = [None] * len(prompts)
        # Prompts that share label tokens go in one request: the common token prefix is the
        # query, each prompt's remainder is an item, so SGLang reuses the prefix cache.
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, prompt in enumerate(prompts):
            groups.setdefault(tuple(prompt.label_token_ids), []).append(index)
        for label_ids, indices in groups.items():
            sequences = [prompts[i].input_ids for i in indices]
            for index, row in zip(indices, self._post(sequences, list(label_ids)), strict=True):
                rows[index] = row
        return BackendResult(
            [row for row in rows if row is not None],
            compute_ms=(time.perf_counter() - started) * 1000,
        )

    def _post(self, sequences: list[list[int]], label_ids: list[int]) -> list[list[float]]:
        shortest = min(len(s) for s in sequences)
        split = 0
        while split < shortest and all(s[split] == sequences[0][split] for s in sequences):
            split += 1
        # Every item must keep at least one token; the label logit is read after the item.
        split = max(0, min(split, shortest - 1))
        body = {
            "model": self.model,
            "query": sequences[0][:split],
            "items": [s[split:] for s in sequences],
            "label_token_ids": label_ids,
            "apply_softmax": False,
        }
        try:
            response = self.client.post("v1/score", json=body)
            response.raise_for_status()
            scores = response.json()["scores"]
            if len(scores) != len(sequences) or any(len(row) != len(label_ids) for row in scores):
                raise ValueError("Unexpected score dimensions")
            rows = []
            for row in scores:
                values = [float(x) for x in row]
                if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
                    raise ValueError("Label probabilities outside [0, 1]")
                if sum(values) > 1.0001:
                    raise ValueError("Label probabilities exceed vocabulary mass")
                rows.append(
                    [max(math.log(x), LOG_PROB_FLOOR) if x > 0 else LOG_PROB_FLOOR for x in values]
                )
            return rows
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError) as exc:
            raise BackendError("SGLang scoring failed or returned invalid probabilities") from exc

    def score_spans(self, prefix_ids: list[int], spans: list[list[int]]) -> BackendResult:
        """Mean token log-prob of each span given the prefix, from /generate prompt log-probs.

        Every span is sent as prefix + span in one batched request; SGLang's radix cache shares the
        prefix across them. The server echoes each scored token's id, and those are checked against
        ours, which also catches a tokenizer or vocabulary mismatch between the two processes.
        Contract-tested against SGLang's documented response shape; not yet exercised live.
        """
        if not spans:
            raise BackendError("No spans to score")
        started = time.perf_counter()
        body = {
            "input_ids": [prefix_ids + span for span in spans],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": len(prefix_ids),
        }
        try:
            response = self.client.post("generate", json=body)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                data = [data]
            if len(data) != len(spans):
                raise ValueError(f"Expected {len(spans)} results, got {len(data)}")
            scores = []
            for item, span in zip(data, spans, strict=True):
                entries = item["meta_info"]["input_token_logprobs"]
                if len(entries) != len(span):
                    raise ValueError(
                        f"Expected {len(span)} prompt log-probs after logprob_start_len, "
                        f"got {len(entries)}"
                    )
                total = 0.0
                for entry, token in zip(entries, span, strict=True):
                    logprob, token_id = entry[0], int(entry[1])
                    if token_id != token:
                        raise ValueError(
                            "Remote tokenization differs from local: token id mismatch in span"
                        )
                    if logprob is None or not math.isfinite(float(logprob)):
                        raise ValueError("Non-finite prompt log-prob")
                    total += float(logprob)
                scores.append(total / len(span))
            return BackendResult([scores], compute_ms=(time.perf_counter() - started) * 1000)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError) as exc:
            raise BackendError("SGLang span scoring failed or returned invalid log-probs") from exc

    def features(self, prefix_ids: list[int], spans: list[list[int]]) -> FeatureResult:
        raise BackendError("The head readout needs hidden states; local backend only")

    def pointer_states(self, prefix_ids, spans, delimiters) -> PointerResult:
        raise BackendError("The pointer readout needs hidden states; local backend only")

    def info(self) -> dict:
        """What the remote server reports it is running. Never raises; see `reachable`."""
        result: dict = {"reachable": False, "model_path": None, "revision": None, "error": None}
        try:
            response = self.client.get("get_server_info")
            if response.status_code == 404:
                # Older servers only expose the model path.
                response = self.client.get("get_model_info")
            response.raise_for_status()
            data = response.json()
            result.update(
                reachable=True,
                model_path=data.get("model_path"),
                revision=data.get("revision"),
            )
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def close(self):
        self.client.close()
