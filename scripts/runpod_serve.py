"""Start the always-on GPU pod that serves the public API with `syn serve`.

The Cloudflare Worker (deploy/cloudflare) sends every call to this pod first and falls back to
the serverless endpoint only when the pod is down, so the endpoint can scale to zero. An
on-demand community-cloud pod costs a fraction of an always-on serverless worker.

The pod clones this repository at --ref (push first), installs the local extra, and serves
SYN_MODEL behind SYN_API_KEY on port 8765, at https://<pod id>-8765.proxy.runpod.net. Weights
are cached on the pod's volume, so a restart doesn't download them again.

    export RUNPOD_API_KEY=...                       # console -> Settings -> API Keys
    uv run python scripts/runpod_serve.py --api-key "$(openssl rand -hex 24)"

Then give the Worker the pod (deploy/cloudflare):

    npx wrangler secret put POD_URL        # https://<pod id>-8765.proxy.runpod.net
    npx wrangler secret put POD_API_KEY    # the --api-key value
"""

import argparse
import json
import os
import subprocess

import httpx

API = "https://rest.runpod.io/v1"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
PORT = 8765
# 24 GB and 16 GB cards that fit Qwen3.5-4B in bfloat16 (9.3 GB of weights), cheapest first.
GPUS = ["NVIDIA RTX A5000", "NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA GeForce RTX 3090"]
START = f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
rm -rf /app && git clone --quiet "$SYN_GIT_URL" /app && cd /app
git fetch --quiet --depth 1 origin "$SYN_GIT_REF" && git checkout --quiet FETCH_HEAD
pip install --quiet ".[local]" hf_transfer
exec syn serve --host 0.0.0.0 --port {PORT}
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def repo_url() -> str:
    url = git("remote", "get-url", "origin")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    return url


def build_request(args: argparse.Namespace) -> dict:
    return {
        "name": args.name,
        "imageName": IMAGE,
        "gpuTypeIds": args.gpu or GPUS,
        "gpuCount": 1,
        "cloudType": args.cloud,
        "computeType": "GPU",
        "containerDiskInGb": 30,
        "volumeInGb": 30,
        "volumeMountPath": "/workspace",
        "supportPublicIp": False,
        "ports": [f"{PORT}/http"],
        "env": {
            "SYN_GIT_URL": repo_url(),
            "SYN_GIT_REF": args.ref or git("rev-parse", "HEAD"),
            # The same settings as the serverless image (deploy/runpod/Dockerfile).
            "SYN_BACKEND": "local",
            "SYN_READOUT": "letters",
            "SYN_MODEL": args.model,
            "SYN_ORDERINGS": "1",
            "SYN_API_KEY": args.api_key,
            "HF_HOME": "/workspace/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        },
        "dockerStartCmd": ["bash", "-lc", START],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api-key", required=True, help="SYN_API_KEY the Worker must present")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--ref", help="commit to serve (default: HEAD, which must be pushed)")
    parser.add_argument("--gpu", action="append", help="GPU type id; repeat for fallbacks")
    parser.add_argument("--cloud", default="COMMUNITY", choices=["SECURE", "COMMUNITY"])
    parser.add_argument("--name", default="sifty-serve")
    parser.add_argument("--dry-run", action="store_true", help="print the request, send nothing")
    args = parser.parse_args(argv)
    body = build_request(args)
    if args.dry_run:
        body["env"]["SYN_API_KEY"] = "<set>"
        print(json.dumps(body, indent=2))
        return
    response = httpx.post(
        f"{API}/pods",
        json=body,
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise SystemExit(f"RunPod refused the pod ({response.status_code}): {response.text}")
    pod = response.json()
    print(
        f"pod {pod['id']} ({pod.get('costPerHr')} $/h): https://{pod['id']}-{PORT}.proxy.runpod.net"
    )
    print("It is ready when GET /health on that URL answers; the first start downloads the model.")


if __name__ == "__main__":
    main()
