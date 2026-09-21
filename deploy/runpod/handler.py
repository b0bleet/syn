"""RunPod serverless worker for the decision scorer.

One GPU worker loads the model once at init and scores one request per job. This reuses the
same Scorer the FastAPI service uses, so numbers measured locally carry over. Everything is
configured through the SYN_* environment variables on the endpoint: SYN_MODEL picks the
backbone (Qwen3-8B fits a 24 GB card in bfloat16), SYN_READOUT and SYN_HEAD_PATH select the
readout, and so on.

Job input is either a ScoreRequest payload or the URL shorthand:

    {"input": {"context": "Win a free iPhone", "question": "Which label best applies?",
               "options": [{"id": "spam", "text": "spam"}, {"id": "ham", "text": "ham"}]}}
    {"input": {"url": "/spam,not+spam/Win+a+free+iPhone"}}

The output is the ScoreResponse dict; `selected_option_id` is the answer. Validation problems
come back as {"error": ...}; backend failures raise, which marks the job FAILED so RunPod
retries it on another worker.
"""

from __future__ import annotations

import runpod
from transformers import AutoTokenizer

from syn.api import load_head
from syn.backends import LocalBackend, resolve_config, revision_commit
from syn.config import Settings
from syn.prompt import PromptBuilder, PromptError
from syn.schema import ScoreRequest
from syn.scoring import Scorer
from syn.shorthand import ShorthandError, build_request, parse

settings = Settings()
config = resolve_config(settings)
commit = revision_commit(config)
tokenizer = AutoTokenizer.from_pretrained(
    settings.model, revision=settings.revision, trust_remote_code=False
)
builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
backend = LocalBackend(settings, config)
head, head_sha = (None, None)
if settings.readout == "head":
    head, head_sha = load_head(settings, config, commit)
scorer = Scorer(settings, builder, backend, commit, head, head_sha)


def handler(job: dict) -> dict:
    payload = job.get("input") or {}
    try:
        if "url" in payload:
            labels, text = parse(payload["url"])
            payload = build_request(labels, text, payload.get("question"))
        request = ScoreRequest.model_validate(payload)
    except (ShorthandError, ValueError) as exc:
        return {"error": str(exc)}
    try:
        return scorer.score(request).model_dump()
    except PromptError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
