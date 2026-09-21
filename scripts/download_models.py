"""Download backbone weights from Hugging Face into the local cache.

    uv run python scripts/download_models.py                       # $SYN_MODEL, else the default
    uv run python scripts/download_models.py Qwen/Qwen3-8B Qwen/Qwen3-4B
    uv run python scripts/download_models.py Qwen/Qwen3-8B@<commit-or-branch>

The project keeps its cache in .cache/huggingface when HF_HOME is set:

    HF_HOME=.cache/huggingface uv run python scripts/download_models.py Qwen/Qwen3-8B

snapshot_download resumes interrupted transfers, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import os

DEFAULT_MODEL = os.environ.get("SYN_MODEL", "Qwen/Qwen3-0.6B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="*", help="repo ids, optionally as model@revision")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for spec in args.models or [DEFAULT_MODEL]:
        model, _, revision = spec.partition("@")
        path = snapshot_download(model, revision=revision or None)
        print(f"{spec} -> {path}")


if __name__ == "__main__":
    main()
