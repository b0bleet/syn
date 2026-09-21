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
import sys
from pathlib import Path

from syn.artifacts import pull, push


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
    args = parser.parse_args(argv)

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
