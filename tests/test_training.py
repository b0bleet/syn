import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from syn.head import AttentionHead
from syn.training import FeatureSet, evaluate_head, head_metrics, train_head

HIDDEN = 24


def write_features(path, n, seed, meta=None):
    """Learnable toy task: the correct option vector is the context mean plus noise."""
    rng = np.random.default_rng(seed)
    arrays, labels = {}, []
    for i in range(n):
        length = int(rng.integers(3, 9))
        ctx = rng.normal(size=(length, HIDDEN)).astype(np.float32)
        count = int(rng.integers(2, 6))
        opt = rng.normal(size=(count, HIDDEN)).astype(np.float32)
        label = int(rng.integers(0, count))
        opt[label] = ctx.mean(0) + 0.1 * rng.normal(size=HIDDEN)
        arrays[f"ctx_{i}"] = ctx.astype(np.float16)
        arrays[f"opt_{i}"] = opt.astype(np.float16)
        labels.append(label)
    meta = {
        "hidden": HIDDEN,
        "model": "toy",
        "revision_commit": "c0ffee",
        "version": "v",
        **(meta or {}),
    }
    np.savez(
        path, labels=np.array(labels, dtype=np.int32), meta=np.array(json.dumps(meta)), **arrays
    )


def test_feature_set_batching_and_control(tmp_path):
    path = tmp_path / "f.npz"
    write_features(path, 5, seed=0)
    fs = FeatureSet(path)
    assert len(fs) == 5 and fs.meta["hidden"] == HIDDEN
    ctx, ctx_mask, opt, opt_mask, labels = fs.batch([0, 1, 2])
    assert ctx.shape[0] == 3 and ctx.shape[2] == HIDDEN
    assert opt.shape == (3, opt_mask.shape[1], HIDDEN)
    assert ctx_mask.dtype == torch.bool and opt_mask.dtype == torch.bool
    assert ctx_mask.sum(1).tolist() == [fs.ctx[i].shape[0] for i in (0, 1, 2)]
    assert opt_mask.sum(1).tolist() == [fs.opt[i].shape[0] for i in (0, 1, 2)]
    assert labels.tolist() == fs.labels[[0, 1, 2]].tolist()
    # Control rotates contexts by one; options and labels stay put.
    _, rotated_mask, _, _, rotated_labels = fs.batch([0, 1, 2], shuffle_context=True)
    assert rotated_mask.sum(1).tolist() == [fs.ctx[i].shape[0] for i in (1, 2, 0)]
    assert rotated_labels.tolist() == labels.tolist()


def test_train_head_learns_and_the_control_collapses(tmp_path):
    random.seed(0)
    train, val, test = (tmp_path / f"{s}.npz" for s in ("train", "val", "test"))
    write_features(train, 600, seed=1)
    write_features(val, 150, seed=2)
    write_features(test, 150, seed=3)
    out = tmp_path / "head.safetensors"
    logs = []
    result = train_head(
        train, val, out, rank=16, epochs=12, batch_size=32, lr=3e-3, log=logs.append
    )
    assert out.exists() and out.with_suffix(".json").exists()
    assert result["best_val_top1"] > 0.85, logs
    report = evaluate_head(out, test)
    assert report["real"]["top1"] > 0.85
    assert report["real"]["examples"] == 150
    low, high = report["real"]["top1_ci95"]
    assert low <= report["real"]["top1"] <= high
    assert 0 <= report["real"]["ece_10_bins"] <= 1
    # Every option set now faces another example's context: nothing to match, so chance.
    control = report["shuffled_context_control"]["top1"]
    assert control < report["real"]["top1"] - 0.3, report
    # Training refuses to overwrite and to mix feature sets.
    with pytest.raises(FileExistsError):
        train_head(train, val, out, epochs=1)
    other = tmp_path / "other.npz"
    write_features(other, 10, seed=4, meta={"model": "different"})
    with pytest.raises(ValueError, match="differ in model"):
        train_head(train, other, tmp_path / "x.safetensors", epochs=1)
    with pytest.raises(ValueError, match="trained on model"):
        evaluate_head(out, other)


def test_head_metrics_without_bootstrap(tmp_path):
    path = tmp_path / "f.npz"
    write_features(path, 8, seed=5)
    torch.manual_seed(0)
    metrics = head_metrics(AttentionHead(HIDDEN, 8), FeatureSet(path), bootstrap=0)
    assert metrics["top1_ci95"] is None and metrics["examples"] == 8
    assert metrics["shuffled_context"] is False
