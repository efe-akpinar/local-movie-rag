"""FAISS index artifacts: build, save and load (index.faiss, records.jsonl, manifest.json)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

INDEX_FILE = "index.faiss"
RECORDS_FILE = "records.jsonl"
MANIFEST_FILE = "manifest.json"


class IndexArtifactError(RuntimeError):
    """The index directory is missing files or its files disagree with each other."""


@dataclass
class MovieIndex:
    index: faiss.Index
    records: list[dict]
    manifest: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value) if not isinstance(value, (list, dict, str)) else False:
        return None
    return str(value)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Exact inner-product index; embeddings must already be L2-normalized (cosine similarity)."""
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save(index_dir: Path, index: faiss.Index, records: list[dict], manifest: dict) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    write_jsonl(index_dir / RECORDS_FILE, records)
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=json_default) + "\n")


def load(index_dir: Path | None = None, expected_embedding_model: str | None = None) -> MovieIndex:
    """Load the artifacts from ``index_dir`` (default: the INDEX_DIR setting) and check they agree."""
    if index_dir is None:
        from movie_rag.config import get_settings

        index_dir = get_settings().index_dir
    index_dir = Path(index_dir)

    missing = [name for name in (INDEX_FILE, RECORDS_FILE, MANIFEST_FILE) if not (index_dir / name).exists()]
    if missing:
        raise IndexArtifactError(f"Index directory {index_dir} is missing: {', '.join(missing)}")

    index = faiss.read_index(str(index_dir / INDEX_FILE))
    records = read_jsonl(index_dir / RECORDS_FILE)
    manifest = json.loads((index_dir / MANIFEST_FILE).read_text())

    if index.ntotal != len(records) or manifest.get("doc_count") != len(records):
        raise IndexArtifactError(
            f"Index has {index.ntotal} vectors, {len(records)} records and a manifest doc_count of "
            f"{manifest.get('doc_count')}; they must match."
        )
    if expected_embedding_model and manifest.get("embedding_model") != expected_embedding_model:
        raise IndexArtifactError(
            f"Index was built with {manifest.get('embedding_model')}, "
            f"but EMBEDDING_MODEL is {expected_embedding_model}."
        )
    return MovieIndex(index=index, records=records, manifest=manifest)
