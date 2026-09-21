import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from syn.head import MASKED, AttentionHead
from syn.training import (
    Augment,
    FeatureSet,
    evaluate_head,
    head_loss,
    head_metrics,
    ranked_probability_score,
    train_head,
    transfer,
)

HIDDEN = 24


def write_features(
    path, n, seed, meta=None, source=None, ordinal=False, extras=True, options=(2, 6)
):
    """Learnable toy task: the correct option vector is the context mean plus noise.

    With `extras`, the file also carries none and distractor vectors, the same in every file
    (fixed seed), as `syn features` writes them. Without `source`, no source column is written,
    which is how files cached before source tags existed look.
    """
    rng = np.random.default_rng(seed)
    arrays, labels = {}, []
    for i in range(n):
        length = int(rng.integers(3, 9))
        ctx = rng.normal(size=(length, HIDDEN)).astype(np.float32)
        count = int(rng.integers(options[0], options[1]))
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
    columns = {}
    if source is not None:
        columns["sources"] = np.array([source] * n, dtype=str)
        columns["ordinal"] = np.array([ordinal] * n, dtype=bool)
    if extras:
        fixed = np.random.default_rng(999)
        columns["none"] = fixed.normal(size=(3, HIDDEN)).astype(np.float16)
        columns["distractors"] = fixed.normal(size=(2, HIDDEN)).astype(np.float16)
    np.savez(
        path,
        labels=np.array(labels, dtype=np.int32),
        meta=np.array(json.dumps(meta)),
        **columns,
        **arrays,
    )


def test_feature_set_batching_and_control(tmp_path):
    path = tmp_path / "f.npz"
    write_features(path, 5, seed=0)
    fs = FeatureSet(path)
    assert len(fs) == 5 and fs.meta["hidden"] == HIDDEN
    ctx, ctx_mask, opt, opt_mask, labels, ordinal = fs.batch([0, 1, 2])
    assert ctx.shape[0] == 3 and ctx.shape[2] == HIDDEN
    assert ordinal.tolist() == [False, False, False]
    assert opt.shape == (3, opt_mask.shape[1], HIDDEN)
    assert ctx_mask.dtype == torch.bool and opt_mask.dtype == torch.bool
    assert ctx_mask.sum(1).tolist() == [fs.ctx[i].shape[0] for i in (0, 1, 2)]
    assert opt_mask.sum(1).tolist() == [fs.opt[i].shape[0] for i in (0, 1, 2)]
    assert labels.tolist() == fs.labels[[0, 1, 2]].tolist()
    # Control rotates contexts by one; options and labels stay put.
    _, rotated_mask, _, _, rotated_labels, _ = fs.batch([0, 1, 2], shuffle_context=True)
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
    assert metrics["shuffled_context"] is False and metrics["temperature"] == 1.0
    assert "by_source" not in metrics  # one source only


def test_files_without_tags_get_a_source_from_the_file_name(tmp_path):
    path = tmp_path / "legacy.npz"
    write_features(path, 4, seed=0, extras=False)
    fs = FeatureSet(path)
    assert fs.sources_present == ["legacy"] and fs.ordinal.tolist() == [False] * 4
    assert fs.none is None and fs.distractors is None
    with pytest.raises(ValueError, match="without none and distractor vectors"):
        fs.batch([0, 1], augment=Augment(p_none=0.5))


def test_augmentation_changes_options_and_labels_as_documented(tmp_path):
    path = tmp_path / "f.npz"
    write_features(path, 12, seed=1, source="s", options=(4, 5))  # four options per row
    fs = FeatureSet(path)
    idx = list(range(12))
    plain = fs.batch(idx)

    # p_none: the answer is removed, a none wording is appended and becomes the answer.
    _, _, opt, opt_mask, labels, ordinal = fs.batch(idx, augment=Augment(p_none=1.0))
    assert opt_mask.sum(1).tolist() == [4] * 12 and labels.tolist() == [3] * 12
    none = torch.from_numpy(fs.none.astype(np.float32))
    for row in range(12):
        assert any(torch.allclose(opt[row, 3], none[j]) for j in range(none.shape[0]))
        original = plain[2][row, plain[4][row]]
        assert not any(torch.allclose(opt[row, k], original) for k in range(4))
    assert not ordinal.any()

    # p_none_distract: a none wording is appended, the answer stays.
    _, _, opt, opt_mask, labels, _ = fs.batch(idx, augment=Augment(p_none_distract=1.0))
    assert opt_mask.sum(1).tolist() == [5] * 12 and labels.tolist() == plain[4].tolist()

    # p_distract: a distractor is appended, the answer stays.
    _, _, opt, opt_mask, labels, _ = fs.batch(idx, augment=Augment(p_distract=1.0))
    assert opt_mask.sum(1).tolist() == [5] * 12 and labels.tolist() == plain[4].tolist()
    distractors = torch.from_numpy(fs.distractors.astype(np.float32))
    assert all(
        any(torch.allclose(opt[row, 4], distractors[j]) for j in range(2)) for row in range(12)
    )

    # Two-option rows never lose their answer; ordinal rows are never touched.
    two = tmp_path / "two.npz"
    write_features(two, 6, seed=2, source="s", options=(2, 3))
    _, _, _, opt_mask, labels, _ = FeatureSet(two).batch(
        list(range(6)), augment=Augment(p_none=1.0)
    )
    assert opt_mask.sum(1).tolist() == [2] * 6
    ladder = tmp_path / "ladder.npz"
    write_features(ladder, 6, seed=3, source="s", ordinal=True, options=(4, 5))
    _, _, _, opt_mask, _, ordinal = FeatureSet(ladder).batch(
        list(range(6)), augment=Augment(p_none_distract=1.0)
    )
    assert opt_mask.sum(1).tolist() == [4] * 6 and ordinal.all()
    with pytest.raises(ValueError, match="sum to at most 1"):
        Augment(0.5, 0.5, 0.5)


def test_ordinal_loss_charges_distance_not_just_misses():
    logits = torch.tensor([[9.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 0.0, 9.0]])
    labels = torch.tensor([0, 0, 0])
    mask = torch.ones(3, 4, dtype=torch.bool)
    rps = ranked_probability_score(logits, labels, mask)
    assert rps[0].item() == pytest.approx(0.0, abs=1e-3)
    assert 0 < rps[1].item() < rps[2].item()  # the far level costs more than the next one
    # Padded positions do not count and the cross-entropy stays the same for both rows.
    padded = torch.cat([logits, torch.full((3, 1), MASKED)], 1)
    padded_mask = torch.cat([mask, torch.zeros(3, 1, dtype=torch.bool)], 1)
    assert torch.allclose(ranked_probability_score(padded, labels, padded_mask), rps, atol=1e-4)
    plain = head_loss(logits, labels, mask, torch.zeros(3, dtype=torch.bool), 1.0)
    ordinal = head_loss(logits, labels, mask, torch.ones(3, dtype=torch.bool), 1.0)
    assert ordinal > plain
    assert head_loss(logits, labels, mask, torch.ones(3, dtype=torch.bool), 0.0) == plain


def test_train_head_fits_and_stores_a_temperature(tmp_path):
    train, val, cal, test = (tmp_path / f"{s}.npz" for s in ("train", "val", "cal", "test"))
    write_features(train, 400, seed=1, source="toy")
    write_features(val, 100, seed=2, source="toy")
    write_features(cal, 100, seed=3, source="toy")
    write_features(test, 100, seed=4, source="toy")
    out = tmp_path / "head.safetensors"
    result = train_head(
        train,
        val,
        out,
        calibration=cal,
        rank=16,
        epochs=6,
        batch_size=32,
        lr=3e-3,
        augment=Augment(p_none=0.1, p_none_distract=0.1, p_distract=0.1),
        log=lambda _: None,
    )
    config = json.loads(out.with_suffix(".json").read_text())
    assert result["temperature"] == config["temperature"] > 0
    assert config["temperature_fit"]["on"] == "calibration"
    assert config["temperature_fit"]["nll_after"] <= config["temperature_fit"]["nll_before"]
    assert config["augment"] == {"p_none": 0.1, "p_none_distract": 0.1, "p_distract": 0.1}
    assert config["sources"] == ["toy"] and config["calibration"] == [str(cal)]
    report = evaluate_head(out, test)
    assert report["temperature"] == config["temperature"]
    assert report["real"]["temperature"] == config["temperature"]
    assert report["real"]["top1"] > 0.8
    # Without a calibration set the temperature is fitted on validation and says so.
    other = tmp_path / "other.safetensors"
    train_head(train, val, other, rank=16, epochs=1, log=lambda _: None)
    assert (
        json.loads(other.with_suffix(".json").read_text())["temperature_fit"]["on"] == "validation"
    )


def test_multi_source_sets_subsets_and_transfer(tmp_path):
    files = {}
    for split, n, seed in (("train", 300, 10), ("validation", 60, 20), ("test", 60, 30)):
        for source in ("alpha", "beta"):
            path = tmp_path / f"{source}-{split}.npz"
            write_features(path, n, seed=seed + len(source), source=source)
            files[(split, source)] = path
    train_paths = [files[("train", "alpha")], files[("train", "beta")]]
    both = FeatureSet(train_paths)
    assert len(both) == 600 and both.sources_present == ["alpha", "beta"]
    assert both.meta["n"] == 600 and both.meta["sources"] == ["alpha", "beta"]
    assert len(both.meta["datasets"]) == 2
    only_beta = both.subset(exclude=["alpha"])
    assert len(only_beta) == 300 and only_beta.sources_present == ["beta"]
    assert len(both.subset(sources=["alpha"])) == 300
    assert len(FeatureSet(train_paths, limit_per_source=50)) == 100
    incompatible = tmp_path / "other.npz"
    write_features(incompatible, 5, seed=1, source="gamma", meta={"model": "different"})
    with pytest.raises(ValueError, match="differs in model"):
        FeatureSet([train_paths[0], incompatible])

    torch.manual_seed(0)
    metrics = head_metrics(AttentionHead(HIDDEN, 8), both, bootstrap=0)
    assert set(metrics["by_source"]) == {"alpha", "beta"}
    assert metrics["by_source"]["alpha"]["examples"] == 300

    out = tmp_path / "transfer"
    report = transfer(
        train_paths,
        [files[("validation", "alpha")], files[("validation", "beta")]],
        [files[("test", "alpha")], files[("test", "beta")]],
        out,
        rank=16,
        epochs=4,
        batch_size=32,
        lr=3e-3,
        log=lambda _: None,
    )
    assert set(report["sources"]) == {"alpha", "beta"}
    for source in ("alpha", "beta"):
        entry = report["sources"][source]
        assert entry["test_examples"] == 60
        assert entry["transfer"]["top1"] > 0.7 and entry["in_domain"]["top1"] > 0.7
        assert entry["control_top1"] < entry["transfer"]["top1"]
        assert (out / f"without-{source}.safetensors").exists()
    assert (out / "all-sources.safetensors").exists()
    assert json.loads((out / "transfer.json").read_text())["general"]["temperature"] > 0
    with pytest.raises(ValueError, match="at least two sources"):
        transfer(train_paths[0], files[("validation", "alpha")], files[("test", "alpha")], out)
