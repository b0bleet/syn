"""RunPod serverless worker for the decision scorer.

One GPU worker loads the model once, on its first job, and scores one request per job. This
reuses the same Scorer the FastAPI service uses, so numbers measured locally carry over.
Everything is configured through the SYN_* environment variables on the endpoint: SYN_MODEL
picks the backbone (Qwen3.5-4B fits a 24 GB card in bfloat16), SYN_READOUT and SYN_HEAD_PATH
select the readout, and so on.

Job input takes one of three forms:

    {"input": {"http": {"method": "GET", "path": "/spam,not+spam/Win+a+free+iPhone"}}}
    {"input": {"context": "Win a free iPhone", "question": "Which label best applies?",
               "options": [{"id": "spam", "text": "spam"}, {"id": "ham", "text": "ham"}]}}
    {"input": {"url": "/spam,not+spam/Win+a+free+iPhone"}}

`http` is what the Cloudflare Worker sends: the request is replayed against the same FastAPI
app `syn serve` runs, and the output is {"status", "headers", "body"}. Callers are
authenticated at the Worker, so leave SYN_API_KEY unset here; the endpoint itself only accepts
the RunPod key. The other two forms return the ScoreResponse dict; problems with them become
the job's error, and backend failures raise, which marks the job FAILED.

The model loads inside the first job rather than at import. A worker that dies while loading at
import restarts in a loop before it ever takes a job, RunPod keeps none of its output, and jobs
wait in the queue until they time out. Loading in a job reports each step as the job's progress
(visible in its status) and turns a failure into the job's error.
"""

from __future__ import annotations

import asyncio
import threading
import traceback

import runpod

from syn.http_job import serve
from syn.prompt import PromptError
from syn.schema import ScoreRequest
from syn.shorthand import ShorthandError, build_request, parse

state: dict = {"scorer": None, "app": None, "error": None}
loading = threading.Lock()


def load(report) -> None:
    """Build the scorer and app once; `report` names each step, and a failure is kept."""
    with loading:
        if state["app"] is not None or state["error"] is not None:
            return
        try:
            report("importing the model libraries")
            from transformers import AutoTokenizer

            from syn.api import create_app, load_head, load_pointer
            from syn.backends import LocalBackend, resolve_config, revision_commit
            from syn.config import Settings
            from syn.prompt import PromptBuilder
            from syn.scoring import Scorer

            settings = Settings()
            report(f"reading the {settings.model} config")
            config = resolve_config(settings)
            commit = revision_commit(config)
            report("loading the tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                settings.model, revision=settings.revision, trust_remote_code=False
            )
            builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
            report("loading the weights onto the GPU")
            backend = LocalBackend(settings, config)
            head, head_sha, head_temperature, pointer = None, None, None, None
            if settings.readout == "head":
                head, head_sha, head_config = load_head(settings, config, commit)
                head_temperature = head_config.get("temperature")
            elif settings.readout == "pointer":
                pointer = load_pointer(settings, config, tokenizer)
            scorer = Scorer(
                settings, builder, backend, commit, head, head_sha, head_temperature, pointer
            )
            state["scorer"], state["app"] = scorer, create_app(settings, scorer)
            report("model loaded")
        except BaseException:  # noqa: BLE001 - any failure, native panics included, must reach the job
            state["error"] = traceback.format_exc()
            print(state["error"], flush=True)


def score(payload: dict) -> dict:
    try:
        if "url" in payload:
            labels, text = parse(payload["url"])
            payload = build_request(labels, text, payload.get("question"))
        request = ScoreRequest.model_validate(payload)
    except (ShorthandError, ValueError) as exc:
        return {"error": str(exc)}
    try:
        return state["scorer"].score(request).model_dump()
    except PromptError as exc:
        return {"error": str(exc)}


async def handler(job: dict) -> dict:
    # RunPod awaits this inside its own event loop, so it must be async; loading and scoring
    # block, so they run in a thread.
    if state["app"] is None:

        def report(step: str) -> None:
            print(f"startup: {step}", flush=True)
            runpod.serverless.progress_update(job, f"startup: {step}")

        await asyncio.to_thread(load, report)
    if state["error"]:
        raise RuntimeError(f"Worker failed to start:\n{state['error'][-4000:]}")
    payload = job.get("input") or {}
    if "http" in payload:
        return await serve(state["app"], payload["http"])
    return await asyncio.to_thread(score, payload)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
