"""The store CLI accepts `--repo` on either side of the subcommand."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hf_store", Path(__file__).parents[1] / "scripts" / "hf_store.py"
)
hf_store = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hf_store)


def test_repo_after_subcommand_reaches_push(monkeypatch, tmp_path, capsys):
    data = tmp_path / "rows"
    data.mkdir()
    seen = {}

    def fake_push(repo, local, target, private=True):
        seen.update(repo=repo, local=local, target=target, private=private)
        return "https://huggingface.co/commit/abc"

    monkeypatch.setattr(hf_store, "push", fake_push)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    hf_store.main(["push", str(data), "--repo", "jolobuild/syn-training"])
    assert seen == {
        "repo": "jolobuild/syn-training",
        "local": data,
        "target": data.as_posix().strip("/"),
        "private": True,
    }
    assert "jolobuild/syn-training" in capsys.readouterr().out


def test_repo_before_subcommand_still_parses(monkeypatch, tmp_path):
    folder = tmp_path / "heads"
    folder.mkdir()
    monkeypatch.setattr(hf_store, "pull", lambda *args: ["heads/run/RESULT.md"])
    monkeypatch.setenv("HF_TOKEN", "test-token")
    hf_store.main(["--repo", "jolobuild/syn-training", "pull", "heads/run", "--to", str(folder)])


def test_missing_token_exits_before_upload(monkeypatch, tmp_path):
    data = tmp_path / "rows"
    data.mkdir()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(hf_store, "get_token", None, raising=False)

    def explode(*_args, **_kwargs):
        raise AssertionError("push should not run without a token")

    monkeypatch.setattr(hf_store, "push", explode)
    # get_token is imported inside _require_token; force the import to see no login.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        hf_store.main(["push", str(data), "--repo", "jolobuild/syn-training"])
