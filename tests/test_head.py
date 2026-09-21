import json

import pytest

torch = pytest.importorskip("torch")

from syn.head import MASKED, AttentionHead, file_sha256


def test_head_shapes_masks_and_permutation_equivariance():
    torch.manual_seed(0)
    head = AttentionHead(16, 8)
    ctx = torch.randn(2, 5, 16)
    ctx_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]).bool()
    opt = torch.randn(2, 3, 16)
    opt_mask = torch.tensor([[1, 1, 0], [1, 1, 1]]).bool()
    logits = head(ctx, ctx_mask, opt, opt_mask)
    assert logits.shape == (2, 3)
    assert logits[0, 2].item() == pytest.approx(MASKED)
    assert torch.isfinite(logits[1]).all()

    # Reordering options reorders logits and changes nothing else.
    perm = torch.tensor([2, 0, 1])
    permuted = head(ctx, ctx_mask, opt[:, perm], opt_mask[:, perm])
    assert torch.allclose(permuted[1], logits[1, perm], atol=1e-5)

    # Masked context tokens cannot influence the result.
    noisy = ctx.clone()
    noisy[0, 3:] = 100.0
    assert torch.allclose(head(noisy, ctx_mask, opt, opt_mask)[0, :2], logits[0, :2], atol=1e-5)
    assert head.parameter_count() == 3 * 16 * 8 + 2 * 2 * 16


def test_head_save_and_load_round_trip(tmp_path):
    torch.manual_seed(1)
    head = AttentionHead(12, 4)
    path = tmp_path / "head.safetensors"
    head.save(path, {"features_meta": {"model": "tiny"}, "best_val_top1": 0.9})
    loaded, config = AttentionHead.load(path)
    assert (config["hidden"], config["rank"]) == (12, 4)
    assert config["features_meta"] == {"model": "tiny"} and config["best_val_top1"] == 0.9
    assert json.loads(path.with_suffix(".json").read_text())["rank"] == 4
    ctx, opt = torch.randn(1, 4, 12), torch.randn(1, 2, 12)
    masks = torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
    assert torch.allclose(loaded(ctx, masks[0], opt, masks[1]), head(ctx, masks[0], opt, masks[1]))
    digest = file_sha256(path)
    assert len(digest) == 64 and digest == file_sha256(path)
