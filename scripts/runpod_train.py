"""Start a RunPod GPU pod that trains the general head and stops itself when done.

    RUNPOD_API_KEY=... uv run python scripts/runpod_train.py --model Qwen/Qwen3-8B --volume-id <id> --wait

The pod pulls a stock PyTorch image, clones this repository at --ref (default: the current
commit, which must be pushed), installs it, and runs deploy/runpod/train_job.py. Results land
on the network volume when one is given (recommended: they outlive the pod, and the serverless
endpoint can load the head from the same volume through SYN_HEAD_PATH), else on the pod's own
disk under /workspace, and optionally in a Hugging Face repo (--upload-repo, with HF_TOKEN set
here so it can be passed through).

    --wait          poll until the pod is gone; the job stops the pod on success and leaves it
                    up on failure so its logs can be read in the RunPod console
    --keep          never stop the pod, even on success
    --dry-run       print the pod request instead of sending it
    --stop POD_ID   terminate a pod
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import httpx

API = "https://rest.runpod.io/v1"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
START = """set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
if [ ! -d /app/.git ]; then git clone --quiet "$SYN_GIT_URL" /app; fi
cd /app
git fetch --quiet --depth 1 origin "$SYN_GIT_REF" && git checkout --quiet FETCH_HEAD
pip install --quiet ".[local,hub]" hf_transfer huggingface_hub
python deploy/runpod/train_job.py
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def repo_url() -> str:
    url = git("remote", "get-url", "origin")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    return url.removesuffix(".git")


def build_request(args: argparse.Namespace) -> dict:
    run = args.run or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    env = {
        "SYN_GIT_URL": args.repo or repo_url(),
        "SYN_GIT_REF": args.ref or git("rev-parse", "HEAD"),
        "SYN_MODEL": args.model,
        "SYN_TRAIN_RUN": run,
        "SYN_TRAIN_LIMIT_PER_SOURCE": str(args.limit_per_source),
        "SYN_TRAIN_EPOCHS": str(args.epochs),
        "HF_HOME": "/runpod-volume/huggingface-cache" if args.volume_id else "/workspace/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if args.revision:
        env["SYN_REVISION"] = args.revision
    if args.sources:
        env["SYN_TRAIN_SOURCES"] = args.sources
    if args.suites:
        env["SYN_TRAIN_SUITES"] = args.suites
    if args.upload_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--upload-repo needs HF_TOKEN in the environment")
        env["SYN_TRAIN_UPLOAD_REPO"] = args.upload_repo
        env["HF_TOKEN"] = token
    if args.keep:
        env["SYN_TRAIN_STOP_POD"] = "0"
    else:
        # The job terminates its own pod on success; on failure it stays up for its logs.
        env["RUNPOD_API_KEY"] = os.environ["RUNPOD_API_KEY"]
    body = {
        "name": f"syn-train-{run}",
        "imageName": IMAGE,
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "containerDiskInGb": args.disk,
        "volumeInGb": 0 if args.volume_id else args.disk,
        "volumeMountPath": "/runpod-volume" if args.volume_id else "/workspace",
        "minRAMPerGPU": args.min_ram,
        "supportPublicIp": False,
        "ports": [],
        "env": env,
        "dockerStartCmd": ["bash", "-lc", START],
    }
    if args.volume_id:
        body["networkVolumeId"] = args.volume_id
    return body


def request(method: str, path: str, **kwargs) -> httpx.Response:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("Set RUNPOD_API_KEY (RunPod console -> Settings -> API Keys)")
    response = httpx.request(
        method, f"{API}{path}", headers={"Authorization": f"Bearer {key}"}, timeout=60, **kwargs
    )
    if response.status_code >= 400:
        raise SystemExit(f"RunPod {method} {path}: HTTP {response.status_code}\n{response.text}")
    return response


def wait(pod_id: str) -> None:
    while True:
        response = httpx.get(
            f"{API}/pods/{pod_id}",
            headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
            timeout=60,
        )
        if response.status_code == 404:
            print("pod is gone: the job finished and stopped it (results are on the volume)")
            return
        pod = response.json() if response.status_code < 400 else {}
        status = pod.get("desiredStatus") or pod.get("status") or f"HTTP {response.status_code}"
        print(f"{datetime.now(UTC).strftime('%H:%M:%S')} {status}", flush=True)
        if status in ("EXITED", "TERMINATED"):
            print("the container exited without stopping the pod: read its logs in the console,")
            print(f"then `python scripts/runpod_train.py --stop {pod_id}`")
            return
        time.sleep(60)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", help="Pin the backbone to a commit (SYN_REVISION)")
    parser.add_argument("--gpu", default="NVIDIA L40S", help="RunPod GPU type id")
    parser.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    parser.add_argument("--disk", type=int, default=80, help="Container disk in GB")
    parser.add_argument("--min-ram", type=int, default=48, help="Minimum host RAM in GB")
    parser.add_argument("--volume-id", help="Network volume mounted at /runpod-volume")
    parser.add_argument("--repo", help="Git URL to clone; defaults to this repo's origin")
    parser.add_argument("--ref", help="Commit, tag, or branch; defaults to HEAD (push it first)")
    parser.add_argument("--run", help="Run name; defaults to a timestamp")
    parser.add_argument("--sources", help="Comma-separated data/ directories to train on")
    parser.add_argument("--suites", help="Comma-separated System One suite directories to import")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--limit-per-source", type=int, default=2000)
    parser.add_argument("--upload-repo", help="Hugging Face repo to upload the run to")
    parser.add_argument("--keep", action="store_true", help="Leave the pod running afterwards")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop", metavar="POD_ID")
    args = parser.parse_args(argv)

    if args.stop:
        request("DELETE", f"/pods/{args.stop}")
        print(f"stopped {args.stop}")
        return
    body = build_request(args)
    if args.dry_run:
        shown = {**body, "env": {**body["env"], "RUNPOD_API_KEY": "<set>", "HF_TOKEN": "<set>"}}
        shown["env"] = {k: v for k, v in shown["env"].items() if k in body["env"]}
        print(json.dumps(shown, indent=2))
        return
    pod = request("POST", "/pods", json=body).json()
    pod_id = pod.get("id")
    print(f"started pod {pod_id} ({body['name']}) on {args.gpu}")
    print("logs: RunPod console -> Pods -> this pod -> Logs")
    if args.wait and pod_id:
        wait(pod_id)


if __name__ == "__main__":
    main(sys.argv[1:])
