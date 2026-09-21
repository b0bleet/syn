"""A Hugging Face Hub repo as the store for training data, cached features, and trained heads.

A training pod has no durable disk unless a network volume is attached. A Hub repo does the
same job with nothing to mount: the training job pulls data and cached features from it before
starting, pushes new features as soon as they are cached and the run directory when training
ends, and the serving endpoint loads a head straight from it:

    SYN_HEAD_PATH=hf://<user>/<repo>/heads/<model>/<run>/all-sources.safetensors

Layout inside the repo (one private model repo is enough):

    data/<source>/{train,validation,calibration,test}.jsonl   rows: yours, or imported suites
    features/<model>/<source>-<split>.npz                      cached features per backbone
    heads/<model>/<run>/                                       checkpoints, transfer.json, RESULT.md
    logs/<run>.log

An `hf://` reference is `hf://<user>/<repo>/<path in repo>[@<revision>]`. Private repos need
HF_TOKEN in the environment wherever they are read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HF_PREFIX = "hf://"


@dataclass(frozen=True)
class HubPath:
    repo_id: str
    path: str
    revision: str | None = None

    def with_path(self, path: str) -> HubPath:
        return HubPath(self.repo_id, path, self.revision)

    def __str__(self) -> str:
        suffix = f"@{self.revision}" if self.revision else ""
        return f"{HF_PREFIX}{self.repo_id}/{self.path}{suffix}"


def is_hub_path(value) -> bool:
    return isinstance(value, str) and value.startswith(HF_PREFIX)


def parse_hub_path(uri: str) -> HubPath:
    if not is_hub_path(uri):
        raise ValueError(f"Not an hf:// reference: {uri!r}")
    rest, _, revision = uri[len(HF_PREFIX) :].partition("@")
    parts = rest.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Expected hf://<user>/<repo>[/<path>][@<revision>], got {uri!r}")
    return HubPath(f"{parts[0]}/{parts[1]}", parts[2] if len(parts) == 3 else "", revision or None)


def token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def download_file(uri: str) -> Path:
    """One file from the Hub into the Hub cache; returns its local path."""
    from huggingface_hub import hf_hub_download

    ref = parse_hub_path(uri)
    if not ref.path:
        raise ValueError(f"{uri!r} names a repo, not a file")
    return Path(hf_hub_download(ref.repo_id, ref.path, revision=ref.revision, token=token()))


def resolve_checkpoint(path_or_uri: str | Path) -> Path:
    """A local checkpoint path as given; an hf:// one fetched together with its JSON sidecar.

    Both files land in the same cache directory, so `with_suffix(".json")` finds the sidecar
    exactly as it does for a local checkpoint.
    """
    if not is_hub_path(path_or_uri):
        return Path(path_or_uri)
    ref = parse_hub_path(str(path_or_uri))
    sidecar = str(Path(ref.path).with_suffix(".json"))
    download_file(str(ref.with_path(sidecar)))
    return download_file(str(path_or_uri))


def pull(
    repo_id: str, path_in_repo: str, local_dir: Path, revision: str | None = None
) -> list[str]:
    """Download `path_in_repo` and everything under it into `local_dir`, keeping the layout.

    Returns the files fetched, relative to `local_dir`; an empty list when the repo or the
    folder does not exist yet, which is the normal first run.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    prefix = path_in_repo.strip("/")
    pattern = f"{prefix}/*" if prefix else "*"
    try:
        snapshot_download(
            repo_id,
            allow_patterns=[pattern],
            local_dir=str(local_dir),
            revision=revision,
            token=token(),
        )
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return []
    root = Path(local_dir) / prefix if prefix else Path(local_dir)
    if not root.exists():
        return []
    return sorted(
        str(p.relative_to(local_dir))
        for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(local_dir).parts
    )


def push(
    repo_id: str,
    local_dir: Path,
    path_in_repo: str,
    message: str | None = None,
    private: bool = True,
) -> str:
    """Upload a directory to `path_in_repo` in one commit, creating the repo if needed."""
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    api.create_repo(repo_id, private=private, exist_ok=True)
    info = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        path_in_repo=path_in_repo.strip("/") or None,
        commit_message=message or f"Add {path_in_repo.strip('/') or 'files'}",
    )
    return getattr(info, "commit_url", None) or str(info)
