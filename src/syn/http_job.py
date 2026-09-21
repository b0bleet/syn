"""Serve one HTTP request carried inside a RunPod job.

The Cloudflare Worker (deploy/cloudflare) checks the caller's key and wraps the request as the
job input {"http": {"method", "path", "headers", "body"}}. The GPU handler replays it against
the same FastAPI app `syn serve` runs, so every route behaves the same in both places and the
Worker holds no API logic.
"""

import httpx
from fastapi import FastAPI

FORWARDED_REQUEST_HEADERS = frozenset({"content-type", "accept"})


def _returned(name: str) -> bool:
    return name == "content-type" or name.startswith("x-")


async def serve(app: FastAPI, request: dict) -> dict:
    """Run `request` through `app` and return {"status", "headers", "body"}."""
    path = request.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"http.path must be an absolute path, got {path!r}")
    headers = {
        k: v
        for k, v in (request.get("headers") or {}).items()
        if k.lower() in FORWARDED_REQUEST_HEADERS
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://syn") as client:
        response = await client.request(
            request.get("method", "GET"), path, headers=headers, content=request.get("body")
        )
    return {
        "status": response.status_code,
        "headers": {k: v for k, v in response.headers.items() if _returned(k)},
        "body": response.text,
    }
