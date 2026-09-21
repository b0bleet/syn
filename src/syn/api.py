import logging
import secrets
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .backends import BackendError, LocalBackend, SGLangBackend, resolve_config, revision_commit
from .config import Settings
from .prompt import HEAD_VERSION, PromptBuilder, PromptError
from .schema import ScoreRequest, ScoreResponse
from .scoring import Scorer
from .shorthand import ShorthandError, build_request, parse, raw_path

logger = logging.getLogger(__name__)


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
    """Load the head checkpoint and refuse one trained on a different backbone or rendering."""
    from .head import AttentionHead, file_sha256

    head, head_config = AttentionHead.load(settings.head_path)
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
    return head, file_sha256(settings.head_path)


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
        if scorer is not None:
            app.state.scorer = scorer
        else:
            from transformers import AutoTokenizer

            # Config first: pins the exact commit and rejects unsupported models before any
            # weights download.
            config = resolve_config(settings)
            commit = revision_commit(config)
            tokenizer = AutoTokenizer.from_pretrained(
                settings.model, revision=settings.revision, trust_remote_code=False
            )
            builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
            if settings.backend == "local":
                backend = LocalBackend(settings, config)
            else:
                backend = SGLangBackend(settings)
                check_remote(settings, backend, commit)
            head, head_sha = (None, None)
            if settings.readout == "head":
                head, head_sha = load_head(settings, config, commit)
            app.state.scorer = Scorer(settings, builder, backend, commit, head, head_sha)
        if not settings.api_key:
            logger.warning("SYN_API_KEY is unset; /v1/score accepts unauthenticated requests")
        try:
            yield
        finally:
            app.state.scorer.backend.close()

    app = FastAPI(title="Syn decision scorer", version="0.6.0", lifespan=lifespan)

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
        fmt: Literal["json", "label"] = Query("json", alias="format"),
    ):
        # The decoded `labels` and `text` above are for the generated docs only. Parsing uses the
        # raw path, because the server has already turned any %2C into a comma.
        try:
            parsed_labels, parsed_text = parse(raw_path(request.scope))
        except ShorthandError as exc:
            raise HTTPException(422, str(exc)) from exc
        payload = build_request(parsed_labels, parsed_text, q)
        try:
            model_request = ScoreRequest.model_validate(payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        response = run(model_request)
        headers = _headers(response)
        if fmt == "label":
            body = response.selected_option_id or f"abstain:{','.join(response.abstain_reasons)}"
            return PlainTextResponse(body + "\n", headers=headers)
        return JSONResponse(response.model_dump(), headers=headers)

    return app
