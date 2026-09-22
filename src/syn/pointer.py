"""Pointer readout: a backbone adapted with low-rank adapters, read by a trained pointer head.

The frozen-feature head (`syn train-head`) never changes the backbone, so it can only re-weight
what the backbone already noticed. This readout adapts the backbone itself and reads the answer
from one sequence:

    prefix (context, question, criteria)  <opt> option 1 </opt>  <opt> option 2 </opt> ...  <decide>

Every option span attends only to the prefix and itself, all spans share the same positions,
and the decide token attends to everything from a fixed position. The head scores option j as
the dot product of a projection of the decide state with a projection of option j's closing
delimiter state, so the logits are exactly invariant to option order by construction, and no
option ever sees another. The delimiters are three of the tokenizer's own rarely used control
tokens; request text can never contain them because `<|name|>` is rewritten before tokenizing.

Training (`syn train-pointer`) runs on a GPU over labelled JSONL rows: listwise cross-entropy,
the ranked probability score on ordinal rows, none-of-the-above and distractor augmentation at
the text level, and an optional anchor loss, a KL from the base model's zero-shot letters
distribution (a `syn evaluate` output on the same rows) that keeps the adapted model from
drifting where it has no evidence. The temperature is fitted on the calibration rows (or
validation) and stored beside the head. On save the adapters are merged into the weights, so
serving needs only the `local` extra:

    SYN_MODEL=<run>/backbone SYN_READOUT=pointer SYN_POINTER_PATH=<run>/pointer.safetensors
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .backends import pointer_layout, pointer_mask
from .cases import extra_cases
from .evaluation import read_jsonl
from .features import DISTRACTOR_TEXTS, NONE_TEXTS, default_source
from .head import file_sha256
from .prompt import POINTER_VERSION, PromptBuilder, PromptError
from .schema import EvalExample, Option
from .training import TEMPERATURE_GRID, Augment, ranked_probability_score, summarize

# Open, close, decide: control tokens Qwen3 defines but never emits in chat.
DELIMITERS = ("<|box_start|>", "<|box_end|>", "<|object_ref_start|>")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NONE_ID = "__none__"
DISTRACTOR_ID = "__distractor__"


def delimiter_ids(tokenizer) -> tuple[int, int, int]:
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in DELIMITERS)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if any(not isinstance(i, int) or i < 0 or i == unknown for i in ids):
        raise ValueError(f"The tokenizer lacks the delimiter tokens {DELIMITERS}")
    return ids  # type: ignore[return-value]


class PointerHead(nn.Module):
    """logit_j = <key(option_j), query(decide)> / sqrt(dim). About 2 x hidden x dim parameters."""

    def __init__(self, hidden: int, dim: int = 256) -> None:
        super().__init__()
        self.hidden, self.dim = hidden, dim
        self.query = nn.Linear(hidden, dim)
        self.key = nn.Linear(hidden, dim)

    def forward(self, decide, options):
        """decide (B, H), options (B, K, H) -> logits (B, K), computed in float32."""
        q = self.query(decide.float())
        k = self.key(options.float())
        return torch.einsum("bkd,bd->bk", k, q) / math.sqrt(self.dim)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: Path, config: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(tensors, str(path))
        sidecar = {"hidden": self.hidden, "dim": self.dim, **config}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))

    @classmethod
    def load(cls, path: Path) -> tuple[PointerHead, dict]:
        path = Path(path)
        config = json.loads(path.with_suffix(".json").read_text())
        head = cls(config["hidden"], config["dim"])
        head.load_state_dict(load_file(str(path)))
        return head.eval(), config


@dataclass(frozen=True)
class PointerReadout:
    """What serving needs: the head, its temperature, its delimiters, and its checksum."""

    head: PointerHead
    config: dict
    temperature: float
    delimiters: tuple[int, int, int]
    sha256: str

    @classmethod
    def load(cls, path: Path, tokenizer, hidden: int) -> PointerReadout:
        head, config = PointerHead.load(path)
        if config["hidden"] != hidden:
            raise ValueError(
                f"Pointer head at {path} was trained on hidden size {config['hidden']}, "
                f"but this backbone has {hidden}"
            )
        ids = delimiter_ids(tokenizer)
        trained = tuple(config.get("delimiters", {}).get("ids", ()))
        if trained != ids:
            raise ValueError(
                f"Pointer head at {path} was trained with delimiter ids {trained}; this "
                f"tokenizer gives {ids}"
            )
        return cls(head, config, float(config.get("temperature", 1.0)), ids, file_sha256(path))


@dataclass
class Encoded:
    ids: list[int]
    positions: list[int]
    segments: list[int]
    ends: list[int]
    decide: int
    label: int
    ordinal: bool
    # The option ids in order, or None when augmentation changed the set (no anchor then).
    option_ids: list[str] | None
    source: str
    uniform: bool = False


def augment_options(
    options: list[Option], label: int, augment: Augment, rng: random.Random
) -> tuple[list[Option], int, bool]:
    """One draw, as in the frozen-feature recipe, at the text level. Returns (options, label, changed)."""
    draw = rng.random()
    none = Option(id=NONE_ID, text=rng.choice(NONE_TEXTS))
    if draw < augment.p_none:
        if len(options) < 3:
            return options, label, False
        kept = [option for i, option in enumerate(options) if i != label]
        return [*kept, none], len(kept), True
    if draw < augment.p_none + augment.p_none_distract:
        return [*options, none], label, True
    if draw < augment.p_none + augment.p_none_distract + augment.p_distract:
        return [*options, Option(id=DISTRACTOR_ID, text=rng.choice(DISTRACTOR_TEXTS))], label, True
    return options, label, False


def encode(
    builder: PromptBuilder,
    example: EvalExample,
    delimiters: tuple[int, int, int],
    augment: Augment | None = None,
    rng: random.Random | None = None,
) -> Encoded:
    request = example.request
    options = list(request.options)
    label = [o.id for o in options].index(example.expected_option_id)
    changed = False
    if augment is not None and augment.active and not example.ordinal and not example.uniform:
        options, label, changed = augment_options(options, label, augment, rng or random.Random())
        if changed:
            request = request.model_copy(update={"options": options})
    cloze = builder.prepare_cloze(request, list_options=False)
    ids, positions, segments, ends, decide = pointer_layout(
        cloze.prefix_ids, cloze.span_ids, delimiters
    )
    return Encoded(
        ids,
        positions,
        segments,
        ends,
        decide,
        label,
        example.ordinal,
        None if changed else [o.id for o in options],
        example.source or "",
        example.uniform,
    )


def batched_logits(decoder, head: PointerHead, batch: list[Encoded], device, dtype):
    """One forward over right-padded rows; a logits tensor over the real options of each."""
    width = max(len(row.ids) for row in batch)
    size = len(batch)
    ids = torch.zeros(size, width, dtype=torch.long, device=device)
    positions = torch.zeros(size, width, dtype=torch.long, device=device)
    mask = torch.full((size, 1, width, width), torch.finfo(dtype).min, dtype=dtype, device=device)
    for b, row in enumerate(batch):
        n = len(row.ids)
        ids[b, :n] = torch.tensor(row.ids, dtype=torch.long, device=device)
        positions[b, :n] = torch.tensor(row.positions, dtype=torch.long, device=device)
        mask[b, 0, :n, :n] = pointer_mask(row.segments, dtype, device)[0, 0]
        padded = torch.arange(n, width, device=device)
        mask[b, 0, padded, padded] = 0  # padding attends to itself, so no row is fully masked
    hidden = decoder(
        input_ids=ids, position_ids=positions, attention_mask=mask, use_cache=False
    ).last_hidden_state
    return [
        head(hidden[b, row.decide][None], hidden[b, row.ends][None])[0]
        for b, row in enumerate(batch)
    ]


def row_loss(
    logits, row: Encoded, ordinal_weight: float, anchor: dict | None, anchor_weight: float
):
    logits = logits.float()
    if row.uniform:
        # The deciding fact is absent. Every option is an equally good answer.
        return -logits.log_softmax(-1).mean()
    label = torch.tensor([row.label], device=logits.device)
    loss = nn.functional.cross_entropy(logits[None], label)
    if row.ordinal and ordinal_weight > 0:
        mask = torch.ones(1, logits.shape[0], dtype=torch.bool, device=logits.device)
        loss = loss + ordinal_weight * ranked_probability_score(logits[None], label, mask)[0]
    # Only rows whose option set augmentation left alone have a matching teacher.
    anchored = anchor is not None and anchor_weight > 0 and row.option_ids is not None
    if anchored and set(anchor) == set(row.option_ids):
        teacher = torch.tensor([anchor[i] for i in row.option_ids], device=logits.device).clamp_min(
            1e-6
        )
        teacher = teacher / teacher.sum()
        # KL(teacher || student): mass the base model put somewhere must be argued away.
        loss = loss + anchor_weight * (teacher * (teacher.log() - logits.log_softmax(-1))).sum()
    return loss


def source_weights(rows: list[EvalExample]) -> list[float]:
    """Per-row weights that give every source the same total, with mean weight 1.

    A source of 2,000 rows otherwise swamps one of 28. Repeating the small source would
    memorize its few wordings, so the loss is scaled instead and every row is still seen.
    """
    counts: dict[str, int] = {}
    for row in rows:
        key = row.source or ""
        counts[key] = counts.get(key, 0) + 1
    raw = [1.0 / counts[row.source or ""] for row in rows]
    scale = len(raw) / sum(raw)
    return [weight * scale for weight in raw]


def hold_out_selection(
    train_rows: list[EvalExample], val_rows: list[EvalExample], seed: int
) -> tuple[list[EvalExample], list[EvalExample], str | None]:
    """Drop one source from training and select the checkpoint on its validation rows.

    The source is the median-sized one that also appears in validation, so selection is not
    the biggest template and not the locked transfer file.
    """
    counts: dict[str, int] = {}
    for row in train_rows:
        key = row.source or ""
        counts[key] = counts.get(key, 0) + 1
    val_sources = {row.source or "" for row in val_rows}
    candidates = sorted(
        (source for source in counts if source in val_sources), key=lambda s: (counts[s], s)
    )
    if len(candidates) < 2:
        return train_rows, val_rows, None
    # The middle of the size ranking, with the seed breaking a tie between the two center sources.
    center = len(candidates) // 2
    pair = candidates[center - 1 : center + 1]
    source = pair[seed % len(pair)]
    train = [row for row in train_rows if (row.source or "") != source]
    select = [row for row in val_rows if (row.source or "") == source]
    if not train or not select:
        return train_rows, val_rows, None
    return train, select, source


def load_rows(paths, limit_per_source: int = 0) -> list[EvalExample]:
    """Labelled rows from one or more JSONL files, tagged with a source when they carry none."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows: list[EvalExample] = []
    per_source: dict[str, int] = {}
    for path in paths:
        path = Path(path)
        fallback = default_source(path)
        for record in read_jsonl(path):
            example = EvalExample.model_validate(record)
            if example.source is None:
                example = example.model_copy(update={"source": fallback})
            source = example.source or fallback
            if limit_per_source and per_source.get(source, 0) >= limit_per_source:
                continue
            per_source[source] = per_source.get(source, 0) + 1
            rows.append(example)
    if not rows:
        raise ValueError(f"No rows in {paths}")
    return rows


def load_anchors(path: Path) -> dict[str, dict[str, float]]:
    """Per example digest, the base model's probability per option id, from `syn evaluate`."""
    anchors = {}
    for row in read_jsonl(Path(path)):
        response = row.get("response")
        if response and "example_sha256" in row:
            anchors[row["example_sha256"]] = {
                score["id"]: float(score["probability"]) for score in response["scores"]
            }
    if not anchors:
        raise ValueError(f"No scored rows in {path}")
    return anchors


@torch.inference_mode()
def score_rows(
    decoder, head, builder, rows, delimiters, device, dtype, batch_size: int, max_tokens: int
):
    """(logits, label, source) per row that fits, for metrics and temperature fitting."""
    decoder.eval()
    head.eval()
    out = []
    for start in range(0, len(rows), batch_size):
        batch = []
        for example in rows[start : start + batch_size]:
            try:
                encoded = encode(builder, example, delimiters)
            except (PromptError, ValueError):
                continue
            if len(encoded.ids) <= max_tokens:
                batch.append(encoded)
        if not batch:
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = batched_logits(decoder, head, batch, device, dtype)
        out += [(z.float().cpu(), row.label, row.source) for z, row in zip(logits, batch)]
    return out


def fit_temperature(scored) -> dict:
    if not scored:
        raise ValueError("No calibration rows")
    width = max(z.shape[0] for z, _, _ in scored)
    padded = torch.full((len(scored), width), -1e9)
    for i, (z, _, _) in enumerate(scored):
        padded[i, : z.shape[0]] = z
    labels = torch.tensor([label for _, label, _ in scored])

    def nll(t: float) -> float:
        return -(padded / t).log_softmax(-1).gather(1, labels[:, None]).mean().item()

    best = min(TEMPERATURE_GRID, key=nll)
    return {
        "temperature": best,
        "examples": len(scored),
        "nll_before": nll(1.0),
        "nll_after": nll(best),
        "at_search_boundary": best in (TEMPERATURE_GRID[0], TEMPERATURE_GRID[-1]),
    }


def train_pointer(
    train,
    validation,
    out_dir,
    *,
    calibration=None,
    test=None,
    indomain=None,
    rank: int = 16,
    head_dim: int = 256,
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 4,
    accumulate: int = 2,
    weight_decay: float = 0.01,
    seed: int = 7,
    augment: Augment | None = None,
    ordinal_weight: float = 0.0,
    anchor=None,
    anchor_weight: float = 0.0,
    max_tokens: int = 2048,
    limit_per_source: int = 0,
    balance_sources: bool = False,
    holdout_selection: bool = False,
    policy_cases: bool = False,
    checkpointing: bool = False,
    settings=None,
    log=print,
) -> dict:
    """Adapt the configured backbone (SYN_MODEL, SYN_REVISION) and train the pointer head.

    Writes `<out_dir>/backbone/` (the merged weights and tokenizer, loadable as SYN_MODEL),
    `<out_dir>/pointer.safetensors` with its JSON sidecar, and `<out_dir>/train.json`.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .backends import resolve_config, revision_commit
    from .config import Settings

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    settings = settings or Settings()
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    augment = augment or Augment()
    config = resolve_config(settings)
    if config.model_type != "qwen3":
        raise ValueError(
            "The pointer readout supports Qwen3 only. Qwen3.5's linear-attention layers "
            "ignore the option mask, so the pointer stays on Qwen3."
        )
    from .artifacts import pretrained_call

    model_id, model_extra = pretrained_call(settings.model, settings.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, **model_extra)
    builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
    delimiters = delimiter_ids(tokenizer)
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    train_rows = load_rows(train, limit_per_source)
    val_rows = load_rows(validation)
    if policy_cases:
        added = extra_cases(train_rows)
        if added:
            log(f"policy cases: {len(added)} extra training rows")
            train_rows = train_rows + added
    selection_source = None
    if holdout_selection:
        train_rows, selection_rows, selection_source = hold_out_selection(
            train_rows, val_rows, seed
        )
        if selection_source is not None:
            log(
                f"checkpoint selection on held-out source {selection_source} ({len(selection_rows)} rows)"
            )
        else:
            selection_rows = val_rows
    else:
        selection_rows = val_rows
    row_weights = source_weights(train_rows) if balance_sources else [1.0] * len(train_rows)
    cal_rows = load_rows(calibration) if calibration else None
    test_rows = load_rows(test) if test else None
    indomain_rows = load_rows(indomain) if indomain else None
    anchors = load_anchors(anchor) if anchor else None
    if anchors is not None:
        from .evaluation import example_digest

        anchored = sum(example_digest(row) in anchors for row in train_rows)
        log(f"anchors cover {anchored} of {len(train_rows)} training rows")

    log(f"loading {settings.model} @ {settings.revision} on {device} ({dtype})")
    lm = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=dtype,
        trust_remote_code=False,
        attn_implementation="sdpa" if device == "cuda" else "eager",
        **model_extra,
    ).to(device)
    peft_model = get_peft_model(
        lm,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="FEATURE_EXTRACTION",
        ),
    )
    if checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm.enable_input_require_grads()
    decoder = lm.model  # the decoder with the adapters injected, without the vocabulary head
    head = PointerHead(config.hidden_size, head_dim).to(device)
    adapters = [p for p in peft_model.parameters() if p.requires_grad]
    log(
        f"train {len(train_rows)} / validation {len(val_rows)} rows; "
        f"{sum(p.numel() for p in adapters) / 1e6:.1f}M adapter and "
        f"{head.parameter_count() / 1e6:.2f}M head parameters"
    )
    optimizer = torch.optim.AdamW(
        [{"params": adapters}, {"params": list(head.parameters())}],
        lr=lr,
        weight_decay=weight_decay,
    )
    batches_per_epoch = math.ceil(len(train_rows) / batch_size)
    steps = max(1, epochs * math.ceil(batches_per_epoch / accumulate))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr, lr], total_steps=steps, pct_start=0.1
    )

    def evaluate(rows, temperature: float = 1.0, bootstrap: int = 0) -> dict:
        scored = score_rows(
            decoder, head, builder, rows, delimiters, device, dtype, batch_size, max_tokens
        )
        return summarize(scored, temperature, bootstrap) if scored else {"top1": 0.0}

    best, best_state, history = -1.0, None, []
    order = list(range(len(train_rows)))
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        peft_model.train()
        head.train()
        rng.shuffle(order)
        total, count, skipped, micro = 0.0, 0, 0, 0

        def step() -> None:
            nn.utils.clip_grad_norm_([*adapters, *head.parameters()], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        for start in range(0, len(order), batch_size):
            batch = []
            for index in order[start : start + batch_size]:
                example = train_rows[index]
                try:
                    encoded = encode(
                        builder, example, delimiters, augment if augment.active else None, rng
                    )
                except (PromptError, ValueError):
                    skipped += 1
                    continue
                if len(encoded.ids) > max_tokens:
                    skipped += 1
                    continue
                batch.append(
                    (
                        encoded,
                        anchors.get(_digest(example)) if anchors else None,
                        row_weights[index],
                    )
                )
            if not batch:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = batched_logits(decoder, head, [row[0] for row in batch], device, dtype)
            losses = [
                weight * row_loss(z, encoded, ordinal_weight, teacher, anchor_weight)
                for z, (encoded, teacher, weight) in zip(logits, batch, strict=True)
            ]
            loss = sum(losses) / sum(row[2] for row in batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in epoch {epoch}; lower the rate")
            (loss / accumulate).backward()
            total += loss.item() * len(batch)
            count += len(batch)
            micro += 1
            if micro % accumulate == 0:
                step()
        if micro % accumulate:
            step()
        val = evaluate(selection_rows)
        in_domain_val = evaluate(val_rows) if selection_source is not None else val
        record = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "rows_skipped": skipped,
            "val_top1": val["top1"],
            "selection_source": selection_source,
            "in_domain_val_top1": in_domain_val["top1"],
            "val_nll": val.get("negative_log_likelihood"),
            "val_ece": val.get("ece_10_bins"),
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(record)
        log(json.dumps(record))
        if val["top1"] > best:
            best = val["top1"]
            best_state = (
                {
                    k: v.detach().cpu().clone()
                    for k, v in peft_model.state_dict().items()
                    if "lora_" in k
                },
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            )
    peft_model.load_state_dict(best_state[0], strict=False)
    head.load_state_dict(best_state[1])

    fit_rows = cal_rows if cal_rows is not None else val_rows
    fit = fit_temperature(
        score_rows(
            decoder, head, builder, fit_rows, delimiters, device, dtype, batch_size, max_tokens
        )
    )
    temperature = fit["temperature"]
    final = evaluate(val_rows, temperature)
    held_out = evaluate(test_rows, temperature) if test_rows else None
    in_domain = evaluate(indomain_rows, temperature) if indomain_rows else None
    log(json.dumps({"temperature": temperature, "val_nll": final.get("negative_log_likelihood")}))
    if held_out is not None:
        log(
            json.dumps(
                {
                    "held_out_top1": held_out.get("top1"),
                    "held_out_nll": held_out.get("negative_log_likelihood"),
                }
            )
        )
    if in_domain is not None:
        log(json.dumps({"in_domain_top1": in_domain.get("top1")}))

    backbone_dir = out_dir / "backbone"
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(backbone_dir, safe_serialization=True)
    tokenizer.save_pretrained(backbone_dir)
    head_path = out_dir / "pointer.safetensors"
    sources = sorted({row.source or "" for row in train_rows})
    head.save(
        head_path,
        {
            "version": POINTER_VERSION,
            "base_model": settings.model,
            "base_revision": settings.revision,
            "base_revision_commit": revision_commit(config),
            "backbone": str(backbone_dir),
            "delimiters": {"tokens": list(DELIMITERS), "ids": list(delimiters)},
            "temperature": temperature,
            "temperature_fit": {
                "on": "calibration" if cal_rows is not None else "validation",
                **fit,
            },
            "rank": rank,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "accumulate": accumulate,
            "weight_decay": weight_decay,
            "seed": seed,
            "augment": {
                "p_none": augment.p_none,
                "p_none_distract": augment.p_none_distract,
                "p_distract": augment.p_distract,
            },
            "ordinal_weight": ordinal_weight,
            "anchor": str(anchor) if anchor else None,
            "anchor_weight": anchor_weight,
            "max_tokens": max_tokens,
            "sources": sources,
            "selection_source": selection_source,
            "balance_sources": balance_sources,
            "best_val_top1": best,
            "validation_at_temperature": {
                "negative_log_likelihood": final.get("negative_log_likelihood"),
                "ece_10_bins": final.get("ece_10_bins"),
            },
            "held_out": held_out,
            "in_domain": in_domain,
            "history": history,
        },
    )
    result = {
        "backbone": str(backbone_dir),
        "pointer": str(head_path),
        "best_val_top1": best,
        "held_out_top1": None if held_out is None else held_out.get("top1"),
        "in_domain_top1": None if in_domain is None else in_domain.get("top1"),
        "temperature": temperature,
        "temperature_fit": fit,
        "epochs": epochs,
        "serve": (
            f"SYN_MODEL={backbone_dir} SYN_READOUT=pointer SYN_POINTER_PATH={head_path} syn serve"
        ),
    }
    (out_dir / "train.json").write_text(json.dumps(result, indent=2))
    return result


def _digest(example: EvalExample) -> str:
    from .evaluation import example_digest

    return example_digest(example)
