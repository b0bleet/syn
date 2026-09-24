import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from .backends import BackendError, LocalBackend, SGLangBackend, resolve_config, revision_commit
from .config import Settings
from .prompt import HEAD_VERSION, PromptBuilder, PromptError
from .schema import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyUsage,
    LabelResult,
    ScoreRequest,
    ScoreResponse,
)
from .scoring import Scorer
from .shorthand import ShorthandError, build_request, check, parse, raw_path
from .systemone import ModelList, SystemOneRequest, SystemOneResponse, models, system_one

logger = logging.getLogger(__name__)

USAGE = """syn: zero-shot text classification

  GET  /<label,label,...>/<text>       /spam,not+spam/Win+a+free+iPhone
  GET  /?labels=<a,b>&text=<text>      /?labels=spam,not+spam&text=Win+a+free+iPhone
  POST / {"input": "text or [texts]", "labels": ["a", "b"]}
  POST /v1/score                       full request and response, see /docs
  POST /v1/systemone                   typesafe-sdk: noul, choice, and score questions

GET answers with the bare label; add ?verbose=1 for JSON.
"""


def _classified(responses: list[ScoreResponse], started: float) -> ClassifyResponse:
    """Compact summary of one or more decisions: label, confidence, and scores per label."""
    return ClassifyResponse(
        model=responses[0].model,
        results=[
            LabelResult(
                label=r.selected_option_id,
                confidence=r.confidence,
                scores={s.id: s.probability for s in r.scores},
                abstain_reasons=r.abstain_reasons,
                ms=r.latency_ms,
            )
            for r in responses
        ],
        usage=ClassifyUsage(
            classifications=len(responses), ms=(time.perf_counter() - started) * 1000
        ),
    )


def _headers(response: ScoreResponse) -> dict[str, str]:
    """Decision summary as headers. Values are percent-encoded; labels may be non-ASCII."""
    return {
        "X-Syn-Selected": quote(response.selected_option_id or ""),
        "X-Syn-Best": quote(response.best_option_id),
        "X-Syn-Confidence": f"{response.confidence:.6f}",
        "X-Syn-Agreement": f"{response.ordering_agreement:.6f}",
        "X-Syn-Abstain-Reasons": ",".join(response.abstain_reasons),
    }


def load_head(settings: Settings, config, commit: str | None):
    """The head, its checksum, and its config; refuses a head from another backbone or rendering."""
    from .artifacts import resolve_checkpoint
    from .head import AttentionHead, file_sha256

    path = resolve_checkpoint(settings.head_path)
    head, head_config = AttentionHead.load(path)
    trained_on = head_config.get("features_meta", {})
    expected = {
        "model": settings.model,
        "version": HEAD_VERSION,
        "hidden": config.hidden_size,
    }
    if commit is not None and trained_on.get("revision_commit") is not None:
        expected["revision_commit"] = commit
    for field, value in expected.items():
        if trained_on.get(field) != value:
            raise ValueError(
                f"Head at {settings.head_path} was trained with {field}={trained_on.get(field)!r}, "
                f"but this service runs {field}={value!r}"
            )
    return head, file_sha256(path), head_config


def load_pointer(settings: Settings, config, tokenizer):
    """The pointer head and its config; refuses one whose hidden size or delimiters differ."""
    from .artifacts import resolve_checkpoint
    from .pointer import PointerReadout

    path = resolve_checkpoint(settings.pointer_path)
    return PointerReadout.load(path, tokenizer, config.hidden_size)


def check_remote(settings: Settings, backend, commit: str | None) -> dict:
    """Refuse to start against an SGLang server that is down or serving a different model.

    The server reports its model path and, if it was launched with --revision, that revision.
    A model path that is a local directory is accepted when it ends with the configured repo
    name. An unpinned remote revision is a warning, not an error: we cannot see which weights it
    loaded, so the operator must pin it on the SGLang side.
    """
    info = backend.info()
    if not info["reachable"]:
        raise RuntimeError(f"SGLang at {settings.sglang_url} is unreachable: {info['error']}")
    remote_model = info.get("model_path")
    repo_name = settings.model.rstrip("/").split("/")[-1]
    if (
        settings.sglang_check_model
        and remote_model
        and remote_model != settings.model
        and not remote_model.rstrip("/").endswith(repo_name)
    ):
        raise RuntimeError(
            f"SGLang serves {remote_model!r} but this service is configured for "
            f"{settings.model!r}. Set SYN_SGLANG_CHECK_MODEL=false to override."
        )
    remote_revision = info.get("revision")
    if remote_revision and remote_revision not in (commit, settings.revision):
        raise RuntimeError(
            f"SGLang was launched with --revision {remote_revision!r}; this service resolved "
            f"{settings.revision!r} to commit {commit}. Both sides must use the same weights."
        )
    if not remote_revision:
        logger.warning(
            "SGLang revision is unpinned. Launch it with --revision %s so both sides use the "
            "same weights; this service cannot verify that from here.",
            commit or settings.revision,
        )
    return info


def load_processor(settings: Settings, model: str, extra: dict, tokenizer):
    """Image inputs for the model when images are enabled, else None (images are refused)."""
    if not settings.images:
        return None
    from transformers import Qwen2VLImageProcessorPil

    from .images import ImageInputs

    # Named explicitly: the automatic loader and the default (fast) processor need torchvision.
    processor = Qwen2VLImageProcessorPil.from_pretrained(model, **extra)
    return ImageInputs(processor, tokenizer)


def create_app(settings: Settings | None = None, scorer: Scorer | None = None) -> FastAPI:
    settings = settings or Settings()

    def require_key(request: Request) -> None:
        """Bearer auth when SYN_API_KEY is set. Constant-time compare; /health is exempt."""
        if not settings.api_key:
            return
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token, settings.api_key):
            raise HTTPException(
                401, "Missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"}
            )

    @asynccontextmanager
    async def lifespan(app):
        if getattr(app.state, "scorer", None) is None:
            from transformers import AutoTokenizer

            from syn.artifacts import pretrained_call

            # Config first: pins the exact commit and rejects unsupported models before any
            # weights download.
            config = resolve_config(settings)
            commit = revision_commit(config)
            model, extra = pretrained_call(settings.model, settings.revision)
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False, **extra)
            processor = load_processor(settings, model, extra, tokenizer)
            builder = PromptBuilder(
                tokenizer,
                settings.max_prompt_tokens,
                settings.prompt_format,
                processor,
                settings.image_max_pixels,
            )
            if settings.backend == "local":
                backend = LocalBackend(settings, config)
            else:
                backend = SGLangBackend(settings)
                check_remote(settings, backend, commit)
            head, head_sha, head_temperature, pointer = None, None, None, None
            if settings.readout == "head":
                head, head_sha, head_config = load_head(settings, config, commit)
                head_temperature = head_config.get("temperature")
            elif settings.readout == "pointer":
                pointer = load_pointer(settings, config, tokenizer)
            app.state.scorer = Scorer(
                settings, builder, backend, commit, head, head_sha, head_temperature, pointer
            )
        if not settings.api_key:
            logger.warning("SYN_API_KEY is unset; classification accepts unauthenticated requests")
        try:
            yield
        finally:
            app.state.scorer.close()

    app = FastAPI(title="Syn decision scorer", version="0.6.0", lifespan=lifespan)
    if scorer is not None:
        # Set now, not in the lifespan, so the app also serves when driven without one, as the
        # RunPod handler does when it replays a Worker's request in-process.
        app.state.scorer = scorer

    def run(request: ScoreRequest) -> ScoreResponse:
        try:
            return app.state.scorer.score(request)
        except PromptError as exc:
            raise HTTPException(422, str(exc)) from exc
        except BackendError as exc:
            logger.exception("Backend scoring failed")
            raise HTTPException(502, "Scoring backend failed") from exc

    @app.get("/health")
    def health():
        """Readiness. Returns 503 when the remote backend cannot be reached."""
        active = getattr(app.state, "scorer", None)
        body = {
            "status": "ready",
            "model": settings.model,
            "revision": settings.revision,
            "revision_commit": active.revision_commit if active else None,
            "backend": settings.backend,
            "prompt_version": active.prompt_version if active else None,
            "readout": settings.readout,
            "head_sha256": active.head_sha256 if active else None,
            "head_temperature": active.head_temperature if active else None,
            "orderings": settings.orderings,
            "auth": "bearer" if settings.api_key else "none",
            "remote_backend_checked": None,
        }
        if settings.backend == "sglang" and active is not None:
            info = active.backend.info()
            body.update(
                remote_backend_checked=True,
                remote_reachable=info["reachable"],
                remote_model_path=info["model_path"],
                remote_revision=info["revision"],
                remote_revision_pinned=bool(info["revision"]),
            )
            if not info["reachable"]:
                body["status"] = "degraded"
                body["remote_error"] = info["error"]
                return JSONResponse(body, status_code=503)
        return body

    @app.post("/v1/score", response_model=ScoreResponse, dependencies=[Depends(require_key)])
    def score(request: ScoreRequest):
        return run(request)

    @app.get("/v1/score", include_in_schema=False)
    def score_method_not_allowed():
        # Without this, the shorthand pattern below would answer GET /v1/score with a confusing
        # "v1 is not a label list" instead of the correct method error.
        raise HTTPException(405, "Use POST for /v1/score, or GET /<labels>/<text> for shorthand")

    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        summary="TypeSafe System One: named noul, choice, and score questions about one state",
        dependencies=[Depends(require_key)],
    )
    def systemone(body: SystemOneRequest, response: Response):
        response.headers["x-typesafe-request-id"] = uuid.uuid4().hex
        try:
            return system_one(run, body, settings.model)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/models", response_model=ModelList, dependencies=[Depends(require_key)])
    def list_models():
        return models(settings.model, settings.readout)

    def to_request(
        labels: list[str], text: str, question: str | None, image: str | None = None
    ) -> ScoreRequest:
        try:
            return ScoreRequest.model_validate(build_request(labels, text, question, image))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def answer(request: ScoreRequest, fmt: str | None, verbose: bool):
        """Bare label by default; verbose=1 for the compact JSON; format=json for everything."""
        started = time.perf_counter()
        response = run(request)
        headers = _headers(response)
        if fmt == "json":
            return JSONResponse(response.model_dump(), headers=headers)
        if verbose and fmt is None:
            return JSONResponse(_classified([response], started).model_dump(), headers=headers)
        body = response.selected_option_id or f"abstain:{','.join(response.abstain_reasons)}"
        return PlainTextResponse(body + "\n", headers=headers)

    @app.get(
        "/",
        summary="Classify text against comma-separated labels (query form)",
        dependencies=[Depends(require_key)],
    )
    def classify_query(
        labels: str | None = Query(None, description="Comma-separated labels"),
        text: str | None = Query(None, description="Text to classify"),
        image: str | None = Query(
            None, description="Image URL to classify instead of, or with, text"
        ),
        q: str | None = Query(None, description="Question to ask; defaults to a generic one"),
        fmt: Literal["json", "label"] | None = Query(None, alias="format"),
        verbose: bool = Query(False, description="JSON with label, confidence, and scores"),
    ):
        if labels is None and text is None and image is None:
            return PlainTextResponse(USAGE)
        if image is not None:
            parsed_labels = [label.strip() for label in (labels or "").split(",") if label.strip()]
            return answer(to_request(parsed_labels, (text or "").strip(), q, image), fmt, verbose)
        try:
            parsed_labels, parsed_text = check((labels or "").split(","), text or "")
        except ShorthandError as exc:
            raise HTTPException(422, str(exc)) from exc
        return answer(to_request(parsed_labels, parsed_text, q), fmt, verbose)

    @app.post(
        "/",
        response_model=ClassifyResponse,
        summary="Classify one text, a batch of up to 32, or an image against labels",
        dependencies=[Depends(require_key)],
    )
    def classify_body(body: ClassifyRequest):
        started = time.perf_counter()
        if body.image is not None:
            request = to_request(body.labels, body.input or "", body.question, body.image)
            return _classified([run(request)], started)
        texts = [body.input] if isinstance(body.input, str) else body.input
        # Validate the whole batch before scoring any of it.
        requests = [to_request(body.labels, text, body.question) for text in texts]
        return _classified([run(request) for request in requests], started)

    # Registered last: its pattern would otherwise shadow the routes above.
    @app.get(
        "/{labels}/{text:path}",
        summary="Classify text against comma-separated labels",
        description=(
            "Shorthand for POST /v1/score. Example: `/spam,not+spam/Win+a+free+iPhone`. "
            "`+` means a space; write a literal comma, plus, or slash as `%2C`, `%2B`, `%2F`."
        ),
        dependencies=[Depends(require_key)],
    )
    def classify(
        request: Request,
        labels: str,
        text: str,
        q: str | None = Query(None, description="Question to ask; defaults to a generic one"),
        fmt: Literal["json", "label"] | None = Query(None, alias="format"),
        verbose: bool = Query(False, description="JSON with label, confidence, and scores"),
    ):
        # The decoded `labels` and `text` above are for the generated docs only. Parsing uses the
        # raw path, because the server has already turned any %2C into a comma.
        try:
            parsed_labels, parsed_text = parse(raw_path(request.scope))
        except ShorthandError as exc:
            raise HTTPException(422, str(exc)) from exc
        return answer(to_request(parsed_labels, parsed_text, q), fmt, verbose)

    return app
