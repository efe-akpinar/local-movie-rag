"""Bi-encoder retrieval and cross-encoder re-ranking, copied from the notebook (Sections 3 and 4)."""

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from movie_rag.index import MovieIndex


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray: ...


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs) -> Sequence[float]: ...


class RerankerError(RuntimeError):
    """The cross-encoder produced scores that cannot be ranked (NaN or infinite)."""


def align_parameters(module) -> int:
    """Copy parameters that sit at unaligned addresses into fresh memory; returns how many moved.

    Weights memory-mapped from a safetensors file can start at an address that is not 64-byte
    aligned. On some BLAS builds (seen with PyTorch's Accelerate backend on macOS) matrix products on
    such tensors return NaN, which would silently disable re-ranking. Copying changes no values.
    """
    moved = 0
    for param in module.parameters():
        if param.data_ptr() % 64:
            param.data = param.data.clone()
            moved += 1
    return moved


def cpu_limit(cgroup_cpu_max: Path = Path("/sys/fs/cgroup/cpu.max")) -> int:
    """CPUs this process may use: os.cpu_count(), capped by the affinity mask and a cgroup v2 CPU quota.

    In a container os.cpu_count() reports the host's CPUs, not the container's limit (a Kubernetes CPU
    limit, docker run --cpus), so the quota cap keeps torch from starting more threads than it gets.
    """
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    try:
        quota, period = cgroup_cpu_max.read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return count


def configure_torch_threads() -> int:
    """Let torch use every CPU the container may use, unless OMP_NUM_THREADS pins the count. Returns it."""
    import torch

    if not os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(cpu_limit())
    return torch.get_num_threads()


def load_bi_encoder(model_name: str, device: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    align_parameters(model)
    return model


def load_cross_encoder(model_name: str, device: str) -> PairScorer:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)
    align_parameters(model.model)
    return model


def encode_documents(encoder: Encoder, texts: list[str], batch_size: int, show_progress: bool = False):
    return encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    ).astype("float32")


class Retriever:
    def __init__(
        self,
        movie_index: MovieIndex,
        bi_encoder: Encoder,
        cross_encoder: PairScorer,
        *,
        k: int = 10,  # bi-encoder candidates sent to the cross-encoder (RERANK_POOL)
        n: int = 5,
        cross_encoder_batch_size: int = 32,
    ):
        self.movie_index = movie_index
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.k = k
        self.n = n
        self.cross_encoder_batch_size = cross_encoder_batch_size

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            return []
        index = self.movie_index.index
        k = max(1, min(int(k or self.k), index.ntotal))
        q_emb = self.bi_encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, idxs = index.search(q_emb, k)
        results = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0], strict=True), start=1):
            if idx < 0:
                continue
            record = dict(self.movie_index.records[int(idx)])
            record["rank"] = rank
            record["score"] = float(score)
            results.append(record)
        return results

    def rerank(self, query: str, candidates: list[dict], n: int | None = None) -> list[dict]:
        if not candidates:
            return []
        n = n or self.n
        pairs = [(query, candidate["document_text"]) for candidate in candidates]
        scores = self.cross_encoder.predict(pairs, batch_size=self.cross_encoder_batch_size, show_progress_bar=False)
        if not np.isfinite(np.asarray(scores, dtype="float64")).all():
            # Sorting NaN scores would silently keep the bi-encoder order; fail loudly instead.
            raise RerankerError("cross-encoder returned non-finite scores")
        reranked = []
        for candidate, score in zip(candidates, scores, strict=True):
            updated = dict(candidate)
            updated["rerank_score"] = float(score)
            reranked.append(updated)
        reranked.sort(key=lambda row: row["rerank_score"], reverse=True)
        for rank, row in enumerate(reranked[:n], start=1):
            row["rerank_rank"] = rank
        return reranked[:n]

    def search(self, query: str, k: int | None = None, n: int | None = None) -> list[dict]:
        return self.rerank(query, self.retrieve(query, k=k), n=n)

    def self_check(self) -> None:
        """Run one query end to end at startup so broken models fail fast instead of serving bad ranks."""
        query = "Who directed Toy Story?"
        if not self.search(query, k=min(self.k, 5), n=1):
            raise RerankerError("self-check retrieved no documents")
