"""Train and evaluate the head on cached features.

Training runs on CPU: the head is under a million parameters and the features are already
computed, so an epoch over a few thousand examples takes seconds.

Three things a plain classification head lacks, all here:

* Calibration in the checkpoint. After the best epoch is chosen, one temperature is fitted on
  held-out rows (the calibration set, or validation when there is none) and saved beside the
  weights, so the served probabilities are calibrated without a separate step.
* An ordinal loss for rows whose options are ordered levels: the ranked probability score on the
  cumulative distribution, which charges for mass far from the true level, not only off it.
* Augmentation with a "None of the above" option and distractors drawn from other rows, so the
  head learns that the right answer can be that nothing offered fits, and that one more
  unrelated option is no reason to change its mind.

A feature set can span several cached files. Every row carries a source tag, so metrics come
per source as well, and `transfer` can hold one source out entirely: train on the rest, test
on it. That number, not in-domain accuracy, says whether a head is general.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import bootstrap_ci, expected_calibration_error
from .head import MASKED, AttentionHead

# Cached files must agree on these before their rows are mixed or a head is applied to them.
COMPATIBILITY_FIELDS = ("model", "revision_commit", "version", "hidden")
# Per-file bookkeeping with no meaning for a merged set.
PER_FILE_META = ("n", "seconds", "dataset", "dataset_sha256", "source")
# A bounded log-spaced search is enough for one parameter; the same grid `syn calibrate` uses.
TEMPERATURE_GRID = sorted(
    {1.0, *(math.exp(math.log(0.05) + i * math.log(400) / 200) for i in range(201))}
)

PathLike = Path | str


@dataclass(frozen=True)
class Augment:
    """Per-row probabilities, applied while training only, never to ordinal rows.

    p_none: the correct option is removed and a "None of the above" wording becomes the answer
    (rows with at least three options). p_none_distract: a none wording is added, but the
    answer stays. p_distract: an unrelated distractor text is added, and the answer stays.
    """

    p_none: float = 0.0
    p_none_distract: float = 0.0
    p_distract: float = 0.0

    def __post_init__(self):
        probabilities = (self.p_none, self.p_none_distract, self.p_distract)
        if any(p < 0 for p in probabilities) or sum(probabilities) > 1:
            raise ValueError("Augmentation probabilities must be non-negative and sum to at most 1")

    @property
    def active(self) -> bool:
        return self.p_none + self.p_none_distract + self.p_distract > 0


class FeatureSet:
    """In-memory view of cached .npz files: per-row context (Lc, H) and option (N, H) arrays.

    Files are checked for the same backbone, revision, and rendering before their rows are
    mixed. `subset` shares the arrays, so holding a source out costs nothing.
    """

    def __init__(self, paths: PathLike | Sequence[PathLike], limit_per_source: int = 0) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("At least one feature file is required")
        self.meta: dict = {}
        self.ctx: list[np.ndarray] = []
        self.opt: list[np.ndarray] = []
        # (M, H) none wordings and (D, H) distractor texts, when the files carry them.
        self.none: np.ndarray | None = None
        self.distractors: np.ndarray | None = None
        labels, ordinal, sources = [], [], []
        per_source: dict[str, int] = {}
        for path in self.paths:
            archive = np.load(path, allow_pickle=False)
            meta = json.loads(str(archive["meta"]))
            if not self.meta:
                self.meta = {k: v for k, v in meta.items() if k not in PER_FILE_META}
                self.meta["datasets"] = []
            else:
                for field in COMPATIBILITY_FIELDS:
                    if meta.get(field) != self.meta.get(field):
                        raise ValueError(f"{path} differs in {field} from {self.paths[0]}")
            self.meta["datasets"].append(meta.get("dataset", str(path)))
            names = set(archive.files)
            file_labels = archive["labels"]
            n = len(file_labels)
            file_ordinal = archive["ordinal"] if "ordinal" in names else np.zeros(n, dtype=bool)
            fallback = meta.get("source") or path.stem
            file_sources = archive["sources"] if "sources" in names else np.array([fallback] * n)
            for name in ("none", "distractors"):
                if name not in names:
                    continue
                vectors = archive[name]
                previous = getattr(self, name)
                if previous is None:
                    setattr(self, name, vectors)
                elif not np.array_equal(vectors, previous):
                    raise ValueError(f"{path} was cached with different {name} vectors")
            for i in range(n):
                source = str(file_sources[i]) or fallback
                if limit_per_source and per_source.get(source, 0) >= limit_per_source:
                    continue
                per_source[source] = per_source.get(source, 0) + 1
                self.ctx.append(archive[f"ctx_{i}"])
                self.opt.append(archive[f"opt_{i}"])
                labels.append(int(file_labels[i]))
                ordinal.append(bool(file_ordinal[i]))
                sources.append(source)
        self.labels = np.array(labels, dtype=np.int64)
        self.ordinal = np.array(ordinal, dtype=bool)
        self.sources = np.array(sources, dtype=str)
        self.meta["n"] = len(self.labels)
        self.meta["sources"] = sorted(per_source)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def sources_present(self) -> list[str]:
        return sorted(set(self.sources.tolist()))

    def subset(
        self, sources: Sequence[str] | None = None, exclude: Sequence[str] | None = None
    ) -> FeatureSet:
        """The rows whose source is in `sources` (all, if None) and not in `exclude`."""
        wanted = None if sources is None else set(sources)
        unwanted = set(exclude or ())
        keep = [
            i
            for i, source in enumerate(self.sources.tolist())
            if (wanted is None or source in wanted) and source not in unwanted
        ]
        view = copy.copy(self)
        view.ctx = [self.ctx[i] for i in keep]
        view.opt = [self.opt[i] for i in keep]
        view.labels = self.labels[keep]
        view.ordinal = self.ordinal[keep]
        view.sources = self.sources[keep]
        view.meta = {**self.meta, "n": len(keep), "sources": view.sources_present}
        return view

    def _augmented(self, options: np.ndarray, label: int, augment: Augment, rng: random.Random):
        """(options, label) after one draw; unchanged when the draw or the row does not qualify."""
        draw = rng.random()
        none = self.none[rng.randrange(self.none.shape[0])][None]
        if draw < augment.p_none:
            if options.shape[0] < 3:
                return options, label
            options = np.concatenate([np.delete(options, label, axis=0), none])
            return options, options.shape[0] - 1
        if draw < augment.p_none + augment.p_none_distract:
            return np.concatenate([options, none]), label
        if draw < augment.p_none + augment.p_none_distract + augment.p_distract:
            distractor = self.distractors[rng.randrange(self.distractors.shape[0])][None]
            return np.concatenate([options, distractor]), label
        return options, label

    def batch(
        self,
        idx: list[int],
        shuffle_context: bool = False,
        augment: Augment | None = None,
        rng: random.Random | None = None,
    ):
        """Padded tensors: ctx (B, Lc, H), ctx_mask, opt (B, N, H), opt_mask, labels, ordinal.

        shuffle_context pairs every example with the next example's context. It is the control
        from the frozen-feature recipe: a head that still scores well is reading option priors,
        not the state. An augmented row loses its ordinal flag, since its options are no longer
        a ladder of levels.
        """
        rng = rng or random.Random(0)
        augmenting = augment is not None and augment.active
        if augmenting and (self.none is None or self.distractors is None):
            raise ValueError(
                "These features were cached without none and distractor vectors; "
                "re-run syn features"
            )
        contexts = [self.ctx[i] for i in idx]
        if shuffle_context:
            contexts = contexts[1:] + contexts[:1]
        rows = []
        for i in idx:
            options, label, ordinal = self.opt[i], int(self.labels[i]), bool(self.ordinal[i])
            if augmenting and not ordinal:
                options, label = self._augmented(options, label, augment, rng)
            rows.append((options, label, ordinal))
        hidden = self.meta["hidden"]
        width_ctx = max(c.shape[0] for c in contexts)
        width_opt = max(options.shape[0] for options, _, _ in rows)
        size = len(idx)
        ctx = torch.zeros(size, width_ctx, hidden)
        ctx_mask = torch.zeros(size, width_ctx, dtype=torch.bool)
        opt = torch.zeros(size, width_opt, hidden)
        opt_mask = torch.zeros(size, width_opt, dtype=torch.bool)
        labels = torch.zeros(size, dtype=torch.int64)
        ordinal = torch.zeros(size, dtype=torch.bool)
        for row, (context, (options, label, is_ordinal)) in enumerate(
            zip(contexts, rows, strict=True)
        ):
            ctx[row, : context.shape[0]] = torch.from_numpy(context.astype(np.float32))
            ctx_mask[row, : context.shape[0]] = True
            opt[row, : options.shape[0]] = torch.from_numpy(options.astype(np.float32))
            opt_mask[row, : options.shape[0]] = True
            labels[row] = label
            ordinal[row] = is_ordinal
        return ctx, ctx_mask, opt, opt_mask, labels, ordinal


def _load(features, limit_per_source: int = 0) -> FeatureSet:
    return features if isinstance(features, FeatureSet) else FeatureSet(features, limit_per_source)


def check_compatible(a: dict, b: dict, what: str) -> None:
    for field in COMPATIBILITY_FIELDS:
        if a.get(field) != b.get(field):
            raise ValueError(f"Train and {what} features differ in {field}")


def ranked_probability_score(logits, labels, opt_mask):
    """Per row: mean over levels of (predicted CDF - true CDF)^2. Zero only when all mass sits on the label."""
    probs = logits.softmax(-1) * opt_mask
    cdf = probs.cumsum(-1)
    levels = torch.arange(logits.shape[1], device=logits.device)[None, :]
    target = (levels >= labels[:, None]).to(probs.dtype)
    squared = ((cdf - target) ** 2) * opt_mask
    return squared.sum(-1) / (opt_mask.sum(-1) - 1).clamp(min=1)


def head_loss(logits, labels, opt_mask, ordinal, ordinal_weight: float):
    """Listwise cross-entropy, plus the ranked probability score on rows marked ordinal."""
    loss = nn.functional.cross_entropy(logits, labels)
    if ordinal_weight > 0 and bool(ordinal.any()):
        loss = (
            loss
            + ordinal_weight * ranked_probability_score(logits, labels, opt_mask)[ordinal].mean()
        )
    return loss


def training_device() -> str:
    """CUDA when present, else CPU. Apple MPS is skipped on purpose: its bfloat16 batch effects
    made head logits depend on batch size, and the head is small enough for a CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def _logits(head: AttentionHead, features: FeatureSet, batch_size: int, shuffle_context: bool):
    """(logits over the row's real options, label, source) for every row, on the CPU."""
    device = training_device()
    head.eval().to(device)
    rows = []
    indices = list(range(len(features)))
    for start in range(0, len(features), batch_size):
        idx = indices[start : start + batch_size]
        ctx, ctx_mask, opt, opt_mask, labels, _ = features.batch(idx, shuffle_context)
        logits = head(ctx.to(device), ctx_mask.to(device), opt.to(device), opt_mask.to(device))
        for row, label in enumerate(labels.tolist()):
            n_real = int(opt_mask[row].sum().item())
            rows.append(
                (logits[row, :n_real].cpu().clone(), label, str(features.sources[idx[row]]))
            )
    return rows


def summarize(rows, temperature: float, bootstrap: int) -> dict:
    """Metrics over (logits, label, source) rows: top-1 with interval, top-3, NLL, Brier, ECE."""
    correct, top3, nll, brier, confidence = [], [], [], [], []
    for logits, label, _ in rows:
        p = (logits / temperature).softmax(-1)
        order = p.argsort(descending=True).tolist()
        correct.append(int(order[0] == label))
        top3.append(int(label in order[:3]))
        nll.append(-math.log(max(p[label].item(), 1e-12)))
        target = torch.zeros_like(p)
        target[label] = 1.0
        brier.append(((p - target) ** 2).sum().item())
        confidence.append(p.max().item())
    n = len(rows)
    return {
        "examples": n,
        "top1": sum(correct) / n,
        "top1_ci95": bootstrap_ci(correct, bootstrap) if bootstrap else None,
        "top3": sum(top3) / n,
        "negative_log_likelihood": sum(nll) / n,
        "multiclass_brier": sum(brier) / n,
        "ece_10_bins": expected_calibration_error(confidence, correct),
    }


def head_metrics(
    head: AttentionHead,
    features: FeatureSet,
    batch_size: int = 64,
    shuffle_context: bool = False,
    bootstrap: int = 1000,
    temperature: float = 1.0,
) -> dict:
    """Top-1 with interval, top-3, NLL, Brier, 10-bin ECE; per source too when there are several."""
    rows = _logits(head, features, batch_size, shuffle_context)
    if not rows:
        raise ValueError("No examples to evaluate")
    report = {
        **summarize(rows, temperature, bootstrap),
        "temperature": temperature,
        "shuffled_context": shuffle_context,
    }
    sources = sorted({source for _, _, source in rows})
    if len(sources) > 1:
        report["by_source"] = {
            source: summarize([r for r in rows if r[2] == source], temperature, bootstrap)
            for source in sources
        }
    return report


def fit_temperature(head: AttentionHead, features: FeatureSet, batch_size: int = 64) -> dict:
    """The temperature that minimises the head's NLL on these rows."""
    rows = _logits(head, features, batch_size, shuffle_context=False)
    if not rows:
        raise ValueError("No calibration examples")
    width = max(logits.shape[0] for logits, _, _ in rows)
    padded = torch.full((len(rows), width), MASKED)
    for i, (logits, _, _) in enumerate(rows):
        padded[i, : logits.shape[0]] = logits
    labels = torch.tensor([label for _, label, _ in rows])

    def nll(t: float) -> float:
        return -(padded / t).log_softmax(-1).gather(1, labels[:, None]).mean().item()

    best = min(TEMPERATURE_GRID, key=nll)
    return {
        "temperature": best,
        "examples": len(rows),
        "nll_before": nll(1.0),
        "nll_after": nll(best),
        "at_search_boundary": best in (TEMPERATURE_GRID[0], TEMPERATURE_GRID[-1]),
    }


def train_head(
    train,
    validation,
    out: PathLike,
    *,
    calibration=None,
    rank: int = 256,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    seed: int = 7,
    augment: Augment | None = None,
    ordinal_weight: float = 1.0,
    limit_per_source: int = 0,
    log=print,
) -> dict:
    """Listwise cross-entropy over each example's options; keeps the best validation epoch.

    `train`, `validation`, and `calibration` are feature files (one or several) or FeatureSets.
    The temperature is fitted on the calibration rows, or on validation when there are none, and
    saved in the checkpoint. Defaults follow the Gemma frozen-feature recipe; 2e-3 diverges on
    some backbones. If the loss spikes in epoch one, lower the rate first.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    random.seed(seed)
    torch.manual_seed(seed)
    train, validation = _load(train, limit_per_source), _load(validation)
    check_compatible(train.meta, validation.meta, "validation")
    if calibration is not None:
        calibration = _load(calibration)
        check_compatible(train.meta, calibration.meta, "calibration")
    if not len(train) or not len(validation):
        raise ValueError("Train and validation sets both need rows")
    augment = augment or Augment()
    if augment.active and (train.none is None or train.distractors is None):
        raise ValueError(
            "Augmentation needs none and distractor vectors; these features were cached without them"
        )
    device = training_device()
    head = AttentionHead(train.meta["hidden"], rank).to(device)
    log(
        f"train {len(train)} / validation {len(validation)} examples from "
        f"{train.sources_present}, hidden {head.hidden}, rank {rank}, "
        f"{head.parameter_count() / 1e6:.2f}M head parameters, on {device}"
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    best, best_state, history = -1.0, None, []
    order = list(range(len(train)))
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        head.train()
        random.shuffle(order)
        total = 0.0
        for start in range(0, len(train), batch_size):
            idx = order[start : start + batch_size]
            ctx, ctx_mask, opt, opt_mask, labels, ordinal = (
                t.to(device)
                for t in train.batch(idx, augment=augment if augment.active else None, rng=rng)
            )
            loss = head_loss(
                head(ctx, ctx_mask, opt, opt_mask), labels, opt_mask, ordinal, ordinal_weight
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in epoch {epoch}; lower the rate")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
        val = head_metrics(head, validation, batch_size, bootstrap=0)
        record = {
            "epoch": epoch,
            "train_loss": total / len(train),
            "val_top1": val["top1"],
            "val_nll": val["negative_log_likelihood"],
            "val_ece": val["ece_10_bins"],
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(record)
        log(json.dumps(record))
        if val["top1"] > best:
            best = val["top1"]
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    fit_on = "calibration" if calibration is not None else "validation"
    fit = fit_temperature(head, calibration if calibration is not None else validation, batch_size)
    temperature = fit["temperature"]
    final = head_metrics(head, validation, batch_size, bootstrap=0, temperature=temperature)
    log(
        json.dumps(
            {
                "temperature": temperature,
                "fit_on": fit_on,
                "val_nll": final["negative_log_likelihood"],
                "val_ece": final["ece_10_bins"],
            }
        )
    )
    head.save(
        out,
        {
            "train": [str(p) for p in train.paths],
            "validation": [str(p) for p in validation.paths],
            "calibration": [str(p) for p in calibration.paths] if calibration is not None else None,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "augment": asdict(augment),
            "ordinal_weight": ordinal_weight,
            "limit_per_source": limit_per_source,
            "sources": train.sources_present,
            "best_val_top1": best,
            "temperature": temperature,
            "temperature_fit": {"on": fit_on, **fit},
            "validation_at_temperature": {
                "negative_log_likelihood": final["negative_log_likelihood"],
                "ece_10_bins": final["ece_10_bins"],
            },
            "features_meta": train.meta,
            "history": history,
        },
    )
    return {
        "best_val_top1": best,
        "checkpoint": str(out),
        "epochs": epochs,
        "temperature": temperature,
        "temperature_fit": {"on": fit_on, **fit},
    }


def evaluate_head(head_path: PathLike, features, batch_size: int = 64) -> dict:
    """Real and shuffled-context metrics at the checkpoint's temperature, after a compatibility check."""
    head, config = AttentionHead.load(Path(head_path))
    features = _load(features)
    trained_on = config.get("features_meta", {})
    for field in COMPATIBILITY_FIELDS:
        if trained_on.get(field) != features.meta.get(field):
            raise ValueError(
                f"Head was trained on {field}={trained_on.get(field)!r}; "
                f"these features have {features.meta.get(field)!r}"
            )
    temperature = float(config.get("temperature", 1.0))
    return {
        "head": str(head_path),
        "features": [str(p) for p in features.paths],
        "temperature": temperature,
        "real": head_metrics(head, features, batch_size, temperature=temperature),
        "shuffled_context_control": head_metrics(
            head, features, batch_size, shuffle_context=True, temperature=temperature
        ),
        "note": (
            "If the control does not collapse toward chance, the head is reading option priors, "
            "not the state."
        ),
    }


def _brief(metrics: dict) -> dict:
    keys = ("top1", "top1_ci95", "negative_log_likelihood", "multiclass_brier", "ece_10_bins")
    return {key: metrics[key] for key in keys}


def transfer(
    train,
    validation,
    test,
    out_dir: PathLike,
    *,
    calibration=None,
    limit_per_source: int = 0,
    log=print,
    **train_kwargs,
) -> dict:
    """Leave-one-source-out: for every source, a head trained without it is tested on it.

    Also trains the head on all sources, so each source's in-domain and transfer numbers come
    from the same test rows. Writes every checkpoint and transfer.json into `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_set = _load(train, limit_per_source)
    val_set, test_set = _load(validation), _load(test)
    cal_set = _load(calibration) if calibration is not None else None
    sources = train_set.sources_present
    if len(sources) < 2:
        raise ValueError("Transfer needs at least two sources in the training features")
    log(f"training on all sources: {sources}")
    general_path = out_dir / "all-sources.safetensors"
    general = train_head(
        train_set, val_set, general_path, calibration=cal_set, log=log, **train_kwargs
    )
    head_all, config_all = AttentionHead.load(general_path)
    report: dict = {"general": general, "sources": {}}
    for source in sources:
        test_rows = test_set.subset(sources=[source])
        val_rows = val_set.subset(exclude=[source])
        if not len(test_rows) or not len(val_rows):
            report["sources"][source] = {
                "skipped": "no test rows for this source, or no validation rows without it"
            }
            continue
        log(f"holding out {source}")
        held_path = out_dir / f"without-{source}.safetensors"
        train_head(
            train_set.subset(exclude=[source]),
            val_rows,
            held_path,
            calibration=cal_set.subset(exclude=[source]) if cal_set is not None else None,
            log=log,
            **train_kwargs,
        )
        head_held, config_held = AttentionHead.load(held_path)
        held_temperature = float(config_held["temperature"])
        transferred = head_metrics(head_held, test_rows, temperature=held_temperature)
        in_domain = head_metrics(head_all, test_rows, temperature=float(config_all["temperature"]))
        control = head_metrics(
            head_held, test_rows, shuffle_context=True, bootstrap=0, temperature=held_temperature
        )
        report["sources"][source] = {
            "test_examples": len(test_rows),
            "in_domain": _brief(in_domain),
            "transfer": _brief(transferred),
            "gap_in_domain_minus_transfer": in_domain["top1"] - transferred["top1"],
            "control_top1": control["top1"],
            "checkpoint": str(held_path),
        }
        log(json.dumps({"source": source, **report["sources"][source]}))
    (out_dir / "transfer.json").write_text(json.dumps(report, indent=2))
    return report
