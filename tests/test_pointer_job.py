"""The pointer training job reads pointer-data/ and the launcher asks the pod to install it."""

import importlib.util
import os
from argparse import Namespace
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_job = _load("train_job", "deploy/runpod/train_job.py")
runpod_train = _load("runpod_train", "scripts/runpod_train.py")


def test_pointer_rows_keep_complete_sources_only(tmp_path, monkeypatch):
    monkeypatch.delenv("SYN_TRAIN_SOURCES", raising=False)
    keep = tmp_path / "decisions"
    keep.mkdir()
    (keep / "train.jsonl").write_text("{}\n")
    (keep / "validation.jsonl").write_text("{}\n")
    (keep / "calibration.jsonl").write_text("{}\n")
    incomplete = tmp_path / "notes"
    incomplete.mkdir()
    (incomplete / "train.jsonl").write_text("{}\n")
    (tmp_path / "transfer-dev.jsonl").write_text("{}\n")

    files = train_job.pointer_rows(tmp_path)
    assert files["train"] == [keep / "train.jsonl"]
    assert files["validation"] == [keep / "validation.jsonl"]
    assert files["calibration"] == [keep / "calibration.jsonl"]


def _args(**overrides):
    values = {
        "run": "t",
        "model": "Qwen/Qwen3-8B",
        "task": "head",
        "revision": None,
        "sources": None,
        "suites": None,
        "hf_repo": None,
        "keep": True,
        "volume_id": None,
        "repo": "https://github.com/example/syn",
        "ref": "abc",
        "gpu": "NVIDIA L40S",
        "cloud": "SECURE",
        "disk": 80,
        "min_ram": 48,
        "limit_per_source": 2000,
        "epochs": 8,
        "anchor": False,
        "ordinal_weight": 0.0,
        "balance_sources": False,
        "holdout_selection": False,
        "policy_cases": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_launcher_marks_a_pointer_pod(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    pointer = runpod_train.build_request(
        _args(task="pointer", hf_repo="jolobuild/syn-training", epochs=None)
    )
    assert pointer["env"]["SYN_TRAIN_TASK"] == "pointer"
    assert pointer["env"]["SYN_TRAIN_POINTER_EPOCHS"] == "2"
    assert pointer["env"]["SYN_TRAIN_ANCHOR"] == "0"
    assert pointer["env"]["SYN_TRAIN_ORDINAL_WEIGHT"] == "0.0"
    assert pointer["env"]["SYN_TRAIN_BALANCE_SOURCES"] == "0"
    assert pointer["env"]["SYN_TRAIN_HOLDOUT_SELECTION"] == "0"
    assert pointer["env"]["SYN_TRAIN_POLICY_CASES"] == "0"
    anchored = runpod_train.build_request(
        _args(task="pointer", hf_repo="jolobuild/syn-training", epochs=None, anchor=True)
    )
    assert anchored["env"]["SYN_TRAIN_ANCHOR"] == "1"
    assert "local,hub,train" in pointer["dockerStartCmd"][-1]
    assert (
        runpod_train.store_prefix("pointer", "jolobuild/syn-training", None)
        == "hf://jolobuild/syn-training/pointers/"
    )
    assert (
        runpod_train.store_prefix("head", "jolobuild/syn-training", None)
        == "hf://jolobuild/syn-training/heads/"
    )
    head = runpod_train.build_request(_args())
    assert head["env"]["SYN_TRAIN_TASK"] == "head"
    assert 'extras="local,hub"' in head["dockerStartCmd"][-1]


@pytest.mark.parametrize(
    "flag,env,value",
    [
        (["--ordinal-weight", "1"], "ORDINAL_WEIGHT", "1.0"),
        (["--balance-sources"], "BALANCE_SOURCES", "1"),
        (["--holdout-selection"], "HOLDOUT_SELECTION", "1"),
        (["--policy-cases"], "POLICY_CASES", "1"),
    ],
)
def test_pointer_cli_changes_one_experiment_at_a_time(flag, env, value, capsys):
    import json

    base = [
        "--task",
        "pointer",
        "--model",
        "Qwen/Qwen3-8B-Base",
        "--keep",
        "--dry-run",
        "--repo",
        "https://github.com/example/syn",
        "--ref",
        "abc",
        "--run",
        "ce-only",
        "--limit-per-source",
        "0",
    ]
    runpod_train.main(base)
    baseline = json.loads(capsys.readouterr().out)["env"]
    runpod_train.main(base + flag)
    experiment = json.loads(capsys.readouterr().out)["env"]
    key = f"SYN_TRAIN_{env}"
    assert {k for k in baseline if baseline[k] != experiment[k]} == {key}
    assert experiment[key] == value
    assert baseline["SYN_TRAIN_POINTER_EPOCHS"] == "2"
    assert baseline["SYN_TRAIN_LIMIT_PER_SOURCE"] == "0"


def test_pointer_job_defaults_reach_trainer(tmp_path, monkeypatch):
    from syn import pointer

    for key in list(os.environ):
        if key.startswith("SYN_TRAIN_"):
            monkeypatch.delenv(key)
    files = {
        split: [tmp_path / f"{split}.jsonl"] for split in ("train", "validation", "calibration")
    }
    for paths in files.values():
        paths[0].write_text("{}\n")
    monkeypatch.setattr(train_job, "prepare_pointer_data", lambda *a: (tmp_path, None))
    monkeypatch.setattr(train_job, "pointer_rows", lambda *a: files)
    captured = {}

    def train(*args, **kwargs):
        captured.update(kwargs)
        args[2].mkdir(parents=True)
        return {"best_val_top1": 0.5, "temperature": 1.0}

    monkeypatch.setattr(pointer, "train_pointer", train)
    train_job.run_pointer(None, tmp_path, "Qwen/Qwen3-8B-Base", "qwen", "test", lambda _: None)
    assert captured["anchor"] is None
    assert captured["anchor_weight"] == 0.0
    assert captured["ordinal_weight"] == 0.0
    assert captured["balance_sources"] is False
    assert captured["holdout_selection"] is False
    assert captured["policy_cases"] is False
    assert captured["limit_per_source"] == 0
