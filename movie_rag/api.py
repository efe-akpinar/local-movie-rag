"""HTTP API: POST /ask, GET /health, GET /ready, GET /metrics (if METRICS_PUBLIC) and an HTML page at /."""

import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from movie_rag import metrics
from movie_rag.config import Settings, get_settings
from movie_rag.logs import configure_logging, log_event
from movie_rag.pipeline import Pipeline, build_pipeline
from movie_rag.ratelimit import SlidingWindowRateLimiter

logger = logging.getLogger("movie_rag.api")

STATIC_DIR = Path(__file__).parent / "static"
KNOWN_ENDPOINTS = {"/", "/ask", "/health", "/ready", "/metrics"}


class AskRequest(BaseModel):
    question: str


class RetrievedTitle(BaseModel):
    title: str
    year: int | None = None
    rerank_score: float


class AskResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[str]
    refused: bool
    retrieved: list[RetrievedTitle]
    timings_ms: dict[str, float]


def client_ip(request: Request, trust_forwarded: bool) -> str:
    """Return the address the rate limiter keys on.

    On Cloud Run the container's socket peer is Google's front end, not the caller, so the socket
    address would put every client in the same bucket. The front end appends the address it accepted
    the connection from to X-Forwarded-For and keeps whatever the caller sent in front of it:

        X-Forwarded-For: <anything the caller sent>, <client ip>

    Only the last entry is written by Google. Earlier entries can be forged to get a fresh window on
    every request, so they are ignored; repeated header lines are joined first for the same reason.
    This holds for the run.app URL that deploy.yml serves. Behind an external Application Load
    Balancer the header ends in "<client ip>, <load balancer ip>" instead, and this would have to
    take the second-to-last entry. With no trusted proxy in front (docker compose, kind) anyone can
    set the header, so TRUST_X_FORWARDED_FOR stays false and the socket address is used.
    """
    if trust_forwarded:
        forwarded = ",".join(request.headers.getlist("x-forwarded-for"))
        last = forwarded.rsplit(",", 1)[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], Pipeline] = build_pipeline,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        start = time.perf_counter()
        app.state.pipeline = await anyio.to_thread.run_sync(pipeline_factory, settings)
        log_event(
            logger,
            "pipeline loaded",
            load_ms=round((time.perf_counter() - start) * 1000, 1),
            documents=app.state.pipeline.retriever.movie_index.size,
            provider=settings.llm_provider,
        )
        yield
        app.state.pipeline = None

    app = FastAPI(title="Movie RAG", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = None
    app.state.settings = settings
    app.state.limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log_event(logger, "unhandled error", logging.ERROR, request_id=request_id, path=request.url.path)
            metrics.observe_error("unhandled")
            response = JSONResponse({"detail": "Internal error", "request_id": request_id}, status_code=500)
        endpoint = request.url.path if request.url.path in KNOWN_ENDPOINTS else "other"
        metrics.observe_request(endpoint, response.status_code, time.perf_counter() - start)
        response.headers["X-Request-ID"] = request_id
        return response

    def ready_pipeline() -> Pipeline:
        pipeline = app.state.pipeline
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Models and index are still loading")
        return pipeline

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        pipeline = ready_pipeline()
        return {
            "status": "ready",
            "documents": pipeline.retriever.movie_index.size,
            "provider": settings.llm_provider,
        }

    if settings.metrics_public:  # otherwise the route does not exist and /metrics is a plain 404

        @app.get("/metrics", include_in_schema=False)
        def prometheus_metrics() -> Response:
            body, content_type = metrics.render()
            return Response(content=body, media_type=content_type)

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        request_id = request.state.request_id
        allowed, retry_after = app.state.limiter.check(client_ip(request, settings.trust_x_forwarded_for))
        if not allowed:
            metrics.observe_error("rate_limited")
            return JSONResponse(
                {"detail": "Too many requests, please slow down.", "request_id": request_id},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be empty")
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                status_code=422,
                detail=f"question must be at most {settings.max_question_chars} characters",
            )

        pipeline = ready_pipeline()
        start = time.perf_counter()
        try:
            result = pipeline.answer(question)
        except Exception:
            logger.exception("pipeline error", extra={"fields": {"request_id": request_id}})
            metrics.observe_error("pipeline")
            return JSONResponse(
                {"detail": "Something went wrong while answering. Please try again.", "request_id": request_id},
                status_code=500,
            )

        metrics.observe_answer(result.timings_ms, result.refused)
        if result.error:
            metrics.observe_error(result.error)
        log_event(
            logger,
            "ask",
            logging.WARNING if result.error else logging.INFO,
            request_id=request_id,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            refused=result.refused,
            error=result.error,
            timings_ms=result.timings_ms,
            question_chars=len(question),
            provider=settings.llm_provider,
        )
        payload = AskResponse(
            request_id=request_id,
            answer=result.answer,
            citations=result.citations,
            refused=result.refused,
            retrieved=[RetrievedTitle(**row) for row in result.retrieved],
            timings_ms=result.timings_ms,
        )
        status = 503 if result.error else 200
        return JSONResponse(payload.model_dump(), status_code=status)

    return app


app = create_app()
