import json

import numpy as np
import pytest

from movie_rag import index as index_module
from movie_rag.index import IndexArtifactError
from tests.conftest import FIXTURE_CORPUS_SIZE


def test_preprocessing_drops_bad_rows_and_duplicates(fixture_corpus):
    df, stats = fixture_corpus
    titles = df["title"].tolist()
    assert len(df) == FIXTURE_CORPUS_SIZE == stats["corpus_size"]
    assert stats["raw_movie_rows"] == 17
    assert "Silent Overview" not in titles  # empty overview
    assert "Bad Id Row" not in titles  # non-numeric id
    assert "Future Film" not in titles  # released after the July 2017 cutoff
    assert titles.count("Jaws") == 1
    jaws = df[df["title"] == "Jaws"].iloc[0]
    assert jaws["movie_id"] == 578  # the duplicate with the longer overview is kept


def test_document_text_format(fixture_corpus):
    df, _ = fixture_corpus
    doc = df[df["title"] == "Toy Story"].iloc[0]["document_text"]
    lines = doc.split("\n")
    assert lines[0] == "Title: Toy Story"
    assert lines[1] == "Year: 1995"
    assert lines[2] == "Genres: Animation, Comedy, Family"
    assert lines[3] == "Director: John Lasseter"
    assert lines[4] == "Cast: Tom Hanks, Tim Allen, Don Rickles, Jim Varney, Wallace Shawn"  # top 5 only
    assert lines[5].startswith("Overview: A pull-string cowboy doll")


def test_index_round_trip(tmp_path, movie_index):
    index_module.save(tmp_path, movie_index.index, movie_index.records, movie_index.manifest)
    loaded = index_module.load(tmp_path, expected_embedding_model="hashing-test-encoder")
    assert loaded.size == FIXTURE_CORPUS_SIZE
    assert [r["title"] for r in loaded.records] == [r["title"] for r in movie_index.records]
    query = np.ones((1, movie_index.index.d), dtype="float32")
    assert np.array_equal(loaded.index.search(query, 3)[1], movie_index.index.search(query, 3)[1])


def test_index_load_rejects_mismatched_artifacts(tmp_path, movie_index):
    index_module.save(tmp_path, movie_index.index, movie_index.records, {"doc_count": 999})
    with pytest.raises(IndexArtifactError, match="must match"):
        index_module.load(tmp_path)


def test_index_load_rejects_other_embedding_model(tmp_path, movie_index):
    index_module.save(tmp_path, movie_index.index, movie_index.records, movie_index.manifest)
    with pytest.raises(IndexArtifactError, match="EMBEDDING_MODEL"):
        index_module.load(tmp_path, expected_embedding_model="BAAI/bge-small-en-v1.5")


def test_index_load_reports_missing_files(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"doc_count": 0}))
    with pytest.raises(IndexArtifactError, match="missing"):
        index_module.load(tmp_path)
