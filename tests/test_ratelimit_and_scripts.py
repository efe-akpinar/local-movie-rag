import importlib.util
import json
import sys
from pathlib import Path

import pytest

from movie_rag import index as index_module
from movie_rag.config import Settings
from movie_rag.ratelimit import SlidingWindowRateLimiter

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_sliding_window_limits_per_key_and_recovers():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(limit=2, window_s=60, clock=clock)
    assert limiter.check("a") == (True, 0)
    clock.now += 10
    assert limiter.check("a") == (True, 0)
    allowed, retry_after = limiter.check("a")
    assert not allowed and retry_after == 50  # the first hit leaves the window in 50 s
    assert limiter.check("b") == (True, 0)  # other clients are unaffected
    clock.now += 50
    assert limiter.check("a") == (True, 0)


def test_rate_limit_zero_disables_the_limiter():
    limiter = SlidingWindowRateLimiter(limit=0)
    assert all(limiter.check("a")[0] for _ in range(1000))


def test_eval_metrics_match_the_notebook_definitions():
    module = load_script("eval_retrieval")
    assert module.reciprocal_rank(["A", "B", "C"], ["C"]) == 1 / 3
    assert module.reciprocal_rank(["A", "B"], ["Z"]) == 0.0
    assert module.recall_at_k(["A", "B", "C"], ["C"], 5) == 1.0
    assert module.recall_at_k(["A", "B", "C"], ["C"], 2) == 0.0


def test_summarize_loadtest(tmp_path):
    header = (
        '"Type","Name","Request Count","Failure Count","Median Response Time","Average Response Time",'
        '"Min Response Time","Max Response Time","Average Content Size","Requests/s","Failures/s",'
        '"50%","66%","75%","80%","90%","95%","98%","99%","99.9%","99.99%","100%"'
    )
    rows = [
        '"POST","/ask",120,2,410,450.5,90,2100,800,0.41,0.007,410,480,520,560,700,900,1200,1500,2100,2100,2100',
        '"","Aggregated",150,2,300,380.2,5,2100,700,0.5,0.007,300,400,460,500,650,850,1100,1400,2100,2100,2100',
    ]
    stats = tmp_path / "run_stats.csv"
    stats.write_text("\n".join([header, *rows]) + "\n")
    table = load_script("summarize_loadtest").summarize(stats)
    lines = table.strip().split("\n")
    assert lines[0] == "| Endpoint | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) | Req/s |"
    assert lines[2] == "| POST /ask | 120 | 2 | 410 | 900 | 1500 | 0.41 |"
    assert lines[3] == "| **Aggregated** | **150** | **2** | **300** | **850** | **1400** | **0.50** |"


def test_upload_only_checks_the_local_index_before_uploading(tmp_path, monkeypatch, movie_index):
    module = load_script("build_index")
    manifest = {"doc_count": movie_index.size, "embedding_model": Settings().embedding_model}
    index_module.save(tmp_path, movie_index.index, movie_index.records, manifest)
    manifest["sha256"] = module.file_hashes(tmp_path)
    (tmp_path / index_module.MANIFEST_FILE).write_text(json.dumps(manifest))
    uploads = []
    monkeypatch.setattr(module, "upload_to_gcs", lambda *args: uploads.append(args))

    def run(*args):
        argv = ["build_index.py", "--upload-only", "--out-dir", str(tmp_path), *args]
        monkeypatch.setattr(sys, "argv", argv)
        return module.main()

    assert run("--gcs-bucket", "b", "--gcs-project", "p", "--expected-count", "12") == 0
    assert uploads == [(tmp_path, "b", "index", "p")]
    assert run("--gcs-bucket", "b") == 1  # 12 documents, but the default expects the full 41,328
    with (tmp_path / index_module.RECORDS_FILE).open("a") as f:
        f.write("\n")  # counts still agree, only the hash changes
    assert run("--gcs-bucket", "b", "--expected-count", "12") == 1
    assert len(uploads) == 1
    with pytest.raises(SystemExit):
        run("--expected-count", "12")  # no bucket
