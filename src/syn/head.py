"""Trainable option-scoring head on frozen backbone features.

A PyTorch port of vinnylarouge/jevlike's AttentionHead, as used by daseinlabs/open-jev. Each
option vector is a query over the context token vectors, and the option's logit is the dot
product of its query with what it attended to. No option ever sees another option, so the head
is permutation-equivariant by construction. About 0.8M parameters at hidden 1024, rank 256.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch import nn

MASKED = -1e9


class AttentionHead(nn.Module):
    def __init__(self, hidden: int, rank: int = 256) -> None:
        super().__init__()
        self.hidden, self.rank = hidden, rank
        self.ctx_norm = nn.LayerNorm(hidden)
        self.opt_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, rank, bias=False)
        self.key = nn.Linear(hidden, rank, bias=False)
        self.value = nn.Linear(hidden, rank, bias=False)

    def forward(self, ctx, ctx_mask, opt, opt_mask):
        """ctx (B, Lc, H), ctx_mask (B, Lc) bool, opt (B, N, H), opt_mask (B, N) bool -> (B, N)."""
        ctx = self.ctx_norm(ctx.float())
        opt = self.opt_norm(opt.float())
        q, k, v = self.query(opt), self.key(ctx), self.value(ctx)
        scores = (q @ k.transpose(1, 2)) / math.sqrt(self.rank)  # (B, N, Lc)
        scores = scores.masked_fill(~ctx_mask[:, None, :], MASKED)
        attended = scores.softmax(-1) @ v  # (B, N, rank)
        logits = (q * attended).sum(-1) / math.sqrt(self.rank)
        return logits.masked_fill(~opt_mask, MASKED)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: Path, config: dict) -> None:
        """Weights as safetensors plus a JSON sidecar with the head shape and training config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(tensors, str(path))
        sidecar = {"hidden": self.hidden, "rank": self.rank, **config}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))

    @classmethod
    def load(cls, path: Path) -> tuple[AttentionHead, dict]:
        path = Path(path)
        config = json.loads(path.with_suffix(".json").read_text())
        head = cls(config["hidden"], config["rank"])
        head.load_state_dict(load_file(str(path)))
        return head.eval(), config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
