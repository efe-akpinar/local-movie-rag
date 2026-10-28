"""End-to-end RAG pipeline: retrieve -> re-rank -> (year guard) -> generate.

``run_query(query)`` keeps the notebook's contract: it returns exactly ``answer`` and
``retrieved_titles`` and never raises.
"""

import logging
import time
from dataclasses import dataclass, field

from movie_rag.config import Settings, get_settings
from movie_rag.generate import (
    REFUSAL_TEXT,
    UNAVAILABLE_TEXT,
    GenerationUnavailable,
    Generator,
    generate_answer,
    is_refusal,
    make_generator,
    mentions_post_cutoff_year,
    parse_citations,
)
from movie_rag.retrieve import Retriever

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    answer: str
    retrieved: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def retrieved_titles(self) -> list[str]:
        return [str(row["title"]) for row in self.retrieved]


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


class Pipeline:
    def __init__(self, retriever: Retriever, generator: Generator, *, k: int = 20, n: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.k = k
        self.n = n

    def answer(self, query) -> PipelineResult:
        start = time.perf_counter()
        query = str(query) if query is not None else ""
        if not query.strip():
            return PipelineResult(answer=REFUSAL_TEXT, refused=True, timings_ms={"total": 0.0})

        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(query, k=self.k)
        t1 = time.perf_counter()
        top_contexts = self.retriever.rerank(query, candidates, n=self.n)
        t2 = time.perf_counter()

        error = None
        if mentions_post_cutoff_year(query):
            answer = REFUSAL_TEXT
        else:
            try:
                answer = generate_answer(self.generator, query, top_contexts)
            except GenerationUnavailable as exc:
                logger.warning("Generation unavailable: %s", exc)
                answer, error = UNAVAILABLE_TEXT, "generation_unavailable"
        t3 = time.perf_counter()

        retrieved = [
            {
                "title": str(row["title"]),
                "year": row.get("release_year"),
                "rerank_score": round(float(row["rerank_score"]), 4),
            }
            for row in top_contexts
        ]
        return PipelineResult(
            answer=answer,
            retrieved=retrieved,
            citations=parse_citations(answer, [row["title"] for row in retrieved]),
            refused=error is None and is_refusal(answer),
            timings_ms={
                "retrieve": _ms(t1 - t0),
                "rerank": _ms(t2 - t1),
                "generate": _ms(t3 - t2),
                "total": _ms(t3 - start),
            },
            error=error,
        )

    def run_query(self, query) -> dict:
        try:
            result = self.answer(query)
            return {
                "answer": str(result.answer or REFUSAL_TEXT),
                "retrieved_titles": result.retrieved_titles,
            }
        except Exception as exc:
            logger.exception("run_query failed")
            return {"answer": f"{REFUSAL_TEXT} Internal error: {type(exc).__name__}", "retrieved_titles": []}


def build_pipeline(settings: Settings | None = None) -> Pipeline:
    """Load the index, both encoders and the generator once."""
    from movie_rag import index as index_module
    from movie_rag.retrieve import load_bi_encoder, load_cross_encoder

    settings = settings or get_settings()
    device = settings.resolved_device()
    movie_index = index_module.load(settings.index_dir, expected_embedding_model=settings.embedding_model)
    retriever = Retriever(
        movie_index,
        load_bi_encoder(settings.embedding_model, device),
        load_cross_encoder(settings.cross_encoder_model, device),
        k=settings.retrieve_k,
        n=settings.rerank_n,
        cross_encoder_batch_size=settings.cross_encoder_batch_size,
    )
    retriever.self_check()
    return Pipeline(retriever, make_generator(settings), k=settings.retrieve_k, n=settings.rerank_n)


_default_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = build_pipeline()
    return _default_pipeline


def run_query(query: str) -> dict:
    """Notebook contract: {"answer": str, "retrieved_titles": list[str]}; never raises."""
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        logger.exception("Pipeline failed to load")
        return {"answer": f"{REFUSAL_TEXT} Internal error: {type(exc).__name__}", "retrieved_titles": []}
    return pipeline.run_query(query)
