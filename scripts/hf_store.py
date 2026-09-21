"""Push data to, or pull results from, the Hugging Face repo the training job uses.

    export HF_TOKEN=...   # huggingface.co/settings/tokens, write access
    uv run python scripts/hf_store.py push data --repo <user>/syn-training
    uv run python scripts/hf_store.py pull heads/Qwen-Qwen3-8B/<run> --repo <user>/syn-training --to runs

`push` uploads a local directory to the same path in the repo (or `--path-in-repo`), so
`push data` publishes every dataset under data/ and `push data/my-suite` just one. `pull`
downloads a repo folder into `--to` (default: the current directory), keeping the layout.
The layout the job expects is documented in src/syn/artifacts.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from syn.artifacts import pull, push


def _repo_first(argv: list[str]) -> list[str]:
    """`--repo` is a top-level option, but the documented command puts it after the subcommand.

    argparse only accepts a parent option before `{push,pull}`, so `push data --repo user/name`
    was rejected as a missing `--repo`. Move it to the front and keep both orders working.
    """
    rest: list[str] = []
    repo: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            rest.extend(argv[i:])
            break
        if arg == "--repo":
            if i + 1 >= len(argv):
                repo = [arg]
                i += 1
                continue
            repo = ["--repo", argv[i + 1]]
            i += 2
            continue
        if arg.startswith("--repo="):
            repo = [arg]
            i += 1
            continue
        rest.append(arg)
        i += 1
    return repo + rest


def _require_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    try:
        from huggingface_hub import get_token
    except ImportError:
        get_token = None
    if get_token is not None and get_token():
        return
    raise SystemExit(
        "No Hugging Face token. Create a write token at https://huggingface.co/settings/tokens "
        "and run: export HF_TOKEN=..."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True, help="<user>/<repo> on huggingface.co")
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("push", help="Upload a local directory")
    up.add_argument("local_dir", type=Path)
    up.add_argument("--path-in-repo", help="Defaults to the local path as given")
    up.add_argument("--public", action="store_true", help="Create the repo public")
    down = sub.add_parser("pull", help="Download a repo folder")
    down.add_argument("path_in_repo")
    down.add_argument("--to", type=Path, default=Path("."))
    down.add_argument("--revision")
    args = parser.parse_args(_repo_first(sys.argv[1:] if argv is None else argv))
    _require_token()

    if args.command == "push":
        if not args.local_dir.is_dir():
            raise SystemExit(f"{args.local_dir} is not a directory")
        target = args.path_in_repo or args.local_dir.as_posix().strip("/")
        url = push(args.repo, args.local_dir, target, private=not args.public)
        print(f"pushed {args.local_dir} -> hf://{args.repo}/{target}\n{url}")
    else:
        files = pull(args.repo, args.path_in_repo, args.to, args.revision)
        if not files:
            raise SystemExit(f"nothing under {args.path_in_repo} in {args.repo}")
        print("\n".join(str(args.to / f) for f in files))


if __name__ == "__main__":
    main(sys.argv[1:])
