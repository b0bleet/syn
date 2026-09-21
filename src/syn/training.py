"""Train and evaluate the head on cached features (jevlike's train.py and eval.py, in PyTorch).

Training runs on CPU: the head is under a million parameters and the features are already
computed, so an epoch over a few thousand examples takes seconds.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import bootstrap_ci
from .head import AttentionHead


class FeatureSet:
    """In-memory view of a cached .npz: per-example context (Lc, H) and option (N, H) arrays."""

    def __init__(self, path: Path) -> None:
        archive = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(archive["meta"]))
        self.labels = archive["labels"]
        self.ctx = [archive[f"ctx_{i}"] for i in range(len(self.labels))]
        self.opt = [archive[f"opt_{i}"] for i in range(len(self.labels))]

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, idx: list[int], shuffle_context: bool = False):
        """Padded tensors: ctx (B, Lc, H), ctx_mask, opt (B, N, H), opt_mask, labels.

        shuffle_context pairs every example with the next example's context. It is the control
        from jevlike: a head that still scores well is reading option priors, not the state.
        """
        contexts = [self.ctx[i] for i in idx]
        if shuffle_context:
            contexts = contexts[1:] + contexts[:1]
        hidden = self.meta["hidden"]
        width_ctx = max(c.shape[0] for c in contexts)
        width_opt = max(self.opt[i].shape[0] for i in idx)
        size = len(idx)
        ctx = torch.zeros(size, width_ctx, hidden)
        ctx_mask = torch.zeros(size, width_ctx, dtype=torch.bool)
        opt = torch.zeros(size, width_opt, hidden)
        opt_mask = torch.zeros(size, width_opt, dtype=torch.bool)
        for row, (context, i) in enumerate(zip(contexts, idx, strict=True)):
            ctx[row, : context.shape[0]] = torch.from_numpy(context.astype(np.float32))
            ctx_mask[row, : context.shape[0]] = True
            options = self.opt[i]
            opt[row, : options.shape[0]] = torch.from_numpy(options.astype(np.float32))
            opt_mask[row, : options.shape[0]] = True
        return ctx, ctx_mask, opt, opt_mask, torch.from_numpy(self.labels[idx].astype(np.int64))


@torch.inference_mode()
def head_metrics(
    head: AttentionHead,
    features: FeatureSet,
    batch_size: int = 64,
    shuffle_context: bool = False,
    bootstrap: int = 1000,
) -> dict:
    """Top-1 with interval, top-3, NLL, Brier, 10-bin ECE on the max probability."""
    head.eval()
    correct, top3, nll, brier, confidence = [], [], [], [], []
    indices = list(range(len(features)))
    for start in range(0, len(features), batch_size):
        idx = indices[start : start + batch_size]
        ctx, ctx_mask, opt, opt_mask, labels = features.batch(idx, shuffle_context)
        probs = head(ctx, ctx_mask, opt, opt_mask).softmax(-1)
        order = probs.argsort(-1, descending=True)
        for row, label in enumerate(labels.tolist()):
            p = probs[row]
            correct.append(int(order[row, 0].item() == label))
            top3.append(int(label in order[row, :3].tolist()))
            nll.append(-math.log(max(p[label].item(), 1e-12)))
            n_real = int(opt_mask[row].sum().item())
            target = torch.zeros(n_real)
            target[label] = 1.0
            brier.append(((p[:n_real] - target) ** 2).sum().item())
            confidence.append(p.max().item())
    n = len(correct)
    ece = 0.0
    for bucket in range(10):
        members = [i for i, c in enumerate(confidence) if min(int(c * 10), 9) == bucket]
        if members:
            ece += abs(sum(confidence[i] - correct[i] for i in members)) / n
    return {
        "examples": n,
        "top1": sum(correct) / n,
        "top1_ci95": bootstrap_ci(correct, bootstrap) if bootstrap else None,
        "top3": sum(top3) / n,
        "negative_log_likelihood": sum(nll) / n,
        "multiclass_brier": sum(brier) / n,
        "ece_10_bins": ece,
        "shuffled_context": shuffle_context,
    }


def train_head(
    train_path: Path,
    validation_path: Path,
    out: Path,
    rank: int = 256,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    seed: int = 7,
    log=print,
) -> dict:
    """Listwise cross-entropy over each example's options; keeps the best validation epoch.

    Defaults follow open-jev's Gemma recipe. jevlike used 2e-3 for Qwen 0.5B; open-jev found it
    diverges on Gemma features. If the loss spikes in epoch one, lower the rate first.
    """
    if out.exists():
        raise FileExistsError(out)
    random.seed(seed)
    torch.manual_seed(seed)
    train, validation = FeatureSet(train_path), FeatureSet(validation_path)
    if train.meta.get("hidden") != validation.meta.get("hidden"):
        raise ValueError("Train and validation features come from different hidden sizes")
    for field in ("model", "revision_commit", "version"):
        if train.meta.get(field) != validation.meta.get(field):
            raise ValueError(f"Train and validation features differ in {field}")
    head = AttentionHead(train.meta["hidden"], rank)
    log(
        f"train {len(train)} / validation {len(validation)} examples, hidden {head.hidden}, "
        f"rank {rank}, {head.parameter_count() / 1e6:.2f}M head parameters"
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, history = -1.0, None, []
    order = list(range(len(train)))
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        head.train()
        random.shuffle(order)
        total = 0.0
        for start in range(0, len(train), batch_size):
            idx = order[start : start + batch_size]
            ctx, ctx_mask, opt, opt_mask, labels = train.batch(idx)
            loss = loss_fn(head(ctx, ctx_mask, opt, opt_mask), labels)
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
    head.save(
        out,
        {
            "train": str(train_path),
            "validation": str(validation_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "best_val_top1": best,
            "features_meta": train.meta,
            "history": history,
        },
    )
    return {"best_val_top1": best, "checkpoint": str(out), "epochs": epochs}


def evaluate_head(head_path: Path, features_path: Path, batch_size: int = 64) -> dict:
    """Real and shuffled-context metrics, plus a check that features match the head."""
    head, config = AttentionHead.load(head_path)
    features = FeatureSet(features_path)
    trained_on = config.get("features_meta", {})
    for field in ("model", "revision_commit", "version", "hidden"):
        if trained_on.get(field) != features.meta.get(field):
            raise ValueError(
                f"Head was trained on {field}={trained_on.get(field)!r}; "
                f"these features have {features.meta.get(field)!r}"
            )
    return {
        "head": str(head_path),
        "features": str(features_path),
        "real": head_metrics(head, features, batch_size),
        "shuffled_context_control": head_metrics(head, features, batch_size, shuffle_context=True),
        "note": (
            "If the control does not collapse toward chance, the head is reading option priors, "
            "not the state."
        ),
    }
