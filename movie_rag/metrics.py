"""Prometheus metrics. Works with one worker and, via PROMETHEUS_MULTIPROC_DIR, with several."""

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 30)

REQUESTS = Counter(
    "movie_rag_requests_total",
    "HTTP requests by endpoint and status code.",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "movie_rag_request_latency_seconds",
    "HTTP request latency by endpoint.",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)
STAGE_LATENCY = Histogram(
    "movie_rag_stage_latency_seconds",
    "Pipeline stage latency (retrieve, rerank, generate).",
    ["stage"],
    buckets=LATENCY_BUCKETS,
)
REFUSALS = Counter("movie_rag_refusals_total", "Answers that were refusals.")
ERRORS = Counter("movie_rag_errors_total", "Failed /ask requests by error type.", ["type"])

STAGES = ("retrieve", "rerank", "generate")


def observe_request(endpoint: str, status: int, latency_s: float) -> None:
    REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_s)


def observe_answer(timings_ms: dict[str, float], refused: bool) -> None:
    for stage in STAGES:
        if stage in timings_ms:
            STAGE_LATENCY.labels(stage=stage).observe(timings_ms[stage] / 1000)
    if refused:
        REFUSALS.inc()


def observe_error(error_type: str) -> None:
    ERRORS.labels(type=error_type).inc()


def render() -> tuple[bytes, str]:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
