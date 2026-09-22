from pathlib import Path

import pytest

from syn import artifacts
from syn.artifacts import HubPath, parse_hub_path, pull, push, resolve_checkpoint


def test_parse_hub_path_forms():
    assert parse_hub_path("hf://u/r") == HubPath("u/r", "")
    assert parse_hub_path("hf://u/r/heads/m/run/all.safetensors") == HubPath(
        "u/r", "heads/m/run/all.safetensors"
    )
    ref = parse_hub_path("hf://u/r/data/x.jsonl@abc123")
    assert ref == HubPath("u/r", "data/x.jsonl", "abc123")
    assert str(ref) == "hf://u/r/data/x.jsonl@abc123"
    assert str(ref.with_path("data/y.jsonl")) == "hf://u/r/data/y.jsonl@abc123"
    for bad in ("u/r", "hf://u", "hf:///r", "hf://u//x"):
        with pytest.raises(ValueError):
            parse_hub_path(bad)


def test_resolve_checkpoint_fetches_the_sidecar_too(tmp_path, monkeypatch):
    assert resolve_checkpoint("runs/head.safetensors") == Path("runs/head.safetensors")
    assert resolve_checkpoint(Path("h.st")) == Path("h.st")
    calls = []

    def fake_download(repo_id, filename, revision=None, token=None):
        calls.append((repo_id, filename, revision, token))
        local = tmp_path / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("x")
        return str(local)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "tok")
    path = resolve_checkpoint("hf://u/r/heads/m/run/all-sources.safetensors@v1")
    assert path == tmp_path / "heads/m/run/all-sources.safetensors"
    assert path.with_suffix(".json").exists()
    assert calls == [
        ("u/r", "heads/m/run/all-sources.json", "v1", "tok"),
        ("u/r", "heads/m/run/all-sources.safetensors", "v1", "tok"),
    ]
    with pytest.raises(ValueError, match="names a repo"):
        artifacts.download_file("hf://u/r")


def test_resolve_pretrained_downloads_a_backbone_directory(tmp_path, monkeypatch):
    from syn.artifacts import pretrained_call, resolve_pretrained

    assert resolve_pretrained("Qwen/Qwen3-0.6B", "main") == ("Qwen/Qwen3-0.6B", "main")

    def fake_snapshot(repo_id, allow_patterns, revision, token):
        folder = tmp_path / "pointers" / "run" / "backbone"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
        assert repo_id == "u/r"
        assert revision == "abc"
        assert token == "tok"
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    monkeypatch.setenv("HF_TOKEN", "tok")
    model, extra = pretrained_call("hf://u/r/pointers/run/backbone@abc", "main")
    assert model == str(tmp_path / "pointers" / "run" / "backbone")
    assert extra == {}
    assert (Path(model) / "config.json").is_file()


def test_pull_lists_fetched_files_and_tolerates_a_missing_repo(tmp_path, monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError

    seen = {}

    def fake_snapshot(repo_id, allow_patterns, local_dir, revision, token):
        seen["args"] = (repo_id, allow_patterns, revision, token)
        if repo_id == "u/missing":
            raise RepositoryNotFoundError("no", response=type("R", (), {"headers": {}, "request": None})())
        for name in ("features/m/a-train.npz", "features/m/a-test.npz", "other/skip.txt"):
            if name.startswith(allow_patterns[0].rstrip("*")):
                target = Path(local_dir) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x")
        (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / ".cache" / "huggingface" / "meta").write_text("")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    files = pull("u/r", "features/m", tmp_path)
    assert files == ["features/m/a-test.npz", "features/m/a-train.npz"]
    assert seen["args"] == ("u/r", ["features/m/*"], None, None)
    assert pull("u/missing", "data", tmp_path) == []
    assert pull("u/r", "nothing-there", tmp_path) == []


def test_push_creates_the_repo_and_uploads_the_folder(tmp_path, monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("api", token))

        def create_repo(self, repo_id, private, exist_ok):
            calls.append(("create", repo_id, private, exist_ok))

        def upload_folder(self, folder_path, repo_id, path_in_repo, commit_message):
            calls.append(("upload", folder_path, repo_id, path_in_repo, commit_message))
            return type("Info", (), {"commit_url": "https://hf.co/commit/1"})()

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setenv("HF_TOKEN", "tok")
    url = push("u/r", tmp_path, "heads/m/run", message="Add run")
    assert url == "https://hf.co/commit/1"
    assert calls == [
        ("api", "tok"),
        ("create", "u/r", True, True),
        ("upload", str(tmp_path), "u/r", "heads/m/run", "Add run"),
    ]
