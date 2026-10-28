"""Offline fixtures: a tiny corpus, a hashing bi-encoder and a word-overlap re-ranker (no downloads)."""

import re
import zlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from movie_rag.api import create_app
from movie_rag.config import Settings
from movie_rag.generate import FakeGenerator
from movie_rag.index import MovieIndex, build_faiss_index
from movie_rag.pipeline import Pipeline
from movie_rag.preprocess import load_and_preprocess_movies, records_from_df
from movie_rag.retrieve import Retriever, encode_documents

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "movies"
FIXTURE_CORPUS_SIZE = 12
_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class HashingEncoder:
    """Deterministic bag-of-words embedding: each token is hashed into one of ``dim`` buckets."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, sentences, normalize_embeddings=True, **kwargs):
        out = np.zeros((len(sentences), self.dim), dtype="float32")
        for row, text in enumerate(sentences):
            for token in tokens(text):
                out[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
        return out


class OverlapScorer:
    """Scores (query, document) pairs by shared words, weighting words in the document's title line."""

    def predict(self, pairs, **kwargs):
        scores = []
        for query, document in pairs:
            q = set(tokens(query))
            title = set(tokens(document.split("\n", 1)[0].removeprefix("Title:")))
            scores.append(2.0 * len(q & title) + len(q & set(tokens(document))) - 0.1 * len(title - q))
        return scores


@pytest.fixture(scope="session")
def fixture_corpus():
    df, stats = load_and_preprocess_movies(FIXTURE_DIR / "movies_metadata.csv", FIXTURE_DIR / "credits.csv")
    return df, stats


@pytest.fixture(scope="session")
def movie_index(fixture_corpus):
    df, _ = fixture_corpus
    encoder = HashingEncoder()
    embeddings = encode_documents(encoder, df["document_text"].tolist(), batch_size=8)
    manifest = {"doc_count": len(df), "embedding_model": "hashing-test-encoder"}
    return MovieIndex(index=build_faiss_index(embeddings), records=records_from_df(df), manifest=manifest)


@pytest.fixture
def retriever(movie_index):
    return Retriever(movie_index, HashingEncoder(), OverlapScorer(), k=20, n=5)


@pytest.fixture
def pipeline(retriever):
    return Pipeline(retriever, FakeGenerator(), k=20, n=5)


@pytest.fixture
def make_client(retriever):
    """Build a TestClient around the offline pipeline; pass a generator or settings overrides."""
    clients = []

    def _make(generator=None, **overrides):
        settings = Settings(**{"llm_provider": "fake", "rate_limit_per_minute": 1000, **overrides})
        pipe = Pipeline(retriever, generator or FakeGenerator(), k=20, n=5)
        client = TestClient(create_app(settings, pipeline_factory=lambda _settings: pipe))
        client.__enter__()  # runs the lifespan, i.e. the startup load
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
