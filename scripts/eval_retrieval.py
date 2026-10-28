"""Retrieval quality gate: Recall@5 and MRR of the re-ranked top 5 on the notebook's 16 answerable queries.

    python scripts/eval_retrieval.py [--index-dir artifacts/index] [--min-recall 0.90] [--pool 10]

--pool is the number of bi-encoder candidates sent to the cross-encoder (default: RERANK_POOL).
Also reports the median re-rank time per query. Exits with status 1 when Recall@5 is below --min-recall.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from movie_rag import index as index_module
from movie_rag.config import get_settings
from movie_rag.retrieve import Retriever, configure_torch_threads, load_bi_encoder, load_cross_encoder
from movie_rag.testset import answerable


def reciprocal_rank(ranked_titles, gold_titles):
    gold = set(gold_titles)
    for rank, title in enumerate(ranked_titles, start=1):
        if title in gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_titles, gold_titles, k):
    return float(any(title in set(gold_titles) for title in ranked_titles[:k]))


def evaluate(retriever: Retriever, k: int, n: int) -> tuple[dict, list[dict]]:
    rows = []
    for item in answerable():
        candidates = retriever.retrieve(item["query"], k=k)
        start = time.perf_counter()
        reranked = retriever.rerank(item["query"], candidates, n=n)
        rerank_ms = (time.perf_counter() - start) * 1000
        titles = [row["title"] for row in reranked]
        rows.append(
            {
                "query": item["query"],
                "category": item["category"],
                "gold_titles": item["gold_titles"],
                "reranked_top5": titles,
                "recall_at_5": recall_at_k(titles, item["gold_titles"], 5),
                "reciprocal_rank": reciprocal_rank(titles[:5], item["gold_titles"]),
                "rerank_ms": round(rerank_ms, 2),
            }
        )
    summary = {
        "queries": len(rows),
        "recall_at_5": sum(r["recall_at_5"] for r in rows) / len(rows),
        "mrr": sum(r["reciprocal_rank"] for r in rows) / len(rows),
        "median_rerank_ms": round(statistics.median(r["rerank_ms"] for r in rows), 2),
        "pool": k,
        "n": n,
    }
    return summary, rows


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--pool", type=int, default=settings.rerank_pool, help="bi-encoder candidates to re-rank")
    parser.add_argument("--json-out", type=Path, help="also write the summary and per-query rows here")
    args = parser.parse_args()

    threads = configure_torch_threads()
    device = settings.resolved_device()
    retriever = Retriever(
        index_module.load(args.index_dir, expected_embedding_model=settings.embedding_model),
        load_bi_encoder(settings.embedding_model, device),
        load_cross_encoder(settings.cross_encoder_model, device),
        k=args.pool,
        n=settings.rerank_n,
        cross_encoder_batch_size=settings.cross_encoder_batch_size,
    )
    retriever.self_check()  # also warms the models up before anything is timed
    summary, rows = evaluate(retriever, args.pool, settings.rerank_n)
    summary.update(device=device, torch_threads=threads)

    for row in rows:
        mark = "hit " if row["recall_at_5"] else "MISS"
        print(f"{mark} RR={row['reciprocal_rank']:.3f}  {row['query']}  ->  {row['reranked_top5']}")
    print(
        f"\nPool {summary['pool']}: Recall@5 = {summary['recall_at_5']:.4f}   MRR = {summary['mrr']:.4f}   "
        f"median re-rank = {summary['median_rerank_ms']:.1f} ms   ({summary['queries']} queries, "
        f"{device}, {threads} torch threads)"
    )
    print(json.dumps(summary))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")

    if summary["recall_at_5"] < args.min_recall:
        print(f"FAIL: Recall@5 {summary['recall_at_5']:.4f} is below --min-recall {args.min_recall}", file=sys.stderr)
        return 1
    print(f"PASS: Recall@5 {summary['recall_at_5']:.4f} >= {args.min_recall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
