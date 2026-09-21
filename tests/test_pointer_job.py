"""The pointer training job reads pointer-data/ and the launcher asks the pod to install it."""

import importlib.util
from argparse import Namespace
from pathlib import Path

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
