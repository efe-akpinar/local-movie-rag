"""End-to-end generation check on the notebook's 20 test queries with the configured LLM_PROVIDER.

    LLM_PROVIDER=groq GROQ_API_KEY=... python scripts/eval_generation.py [--index-dir artifacts/index]

Reports refusals on the 4 unanswerable queries and, for the 16 answerable ones, whether the expected
title is cited as [Title]. Saves everything to results/generation_<provider>_<timestamp>.json.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from movie_rag.config import get_settings
from movie_rag.generate import is_refusal
from movie_rag.pipeline import build_pipeline
from movie_rag.testset import TEST_SET


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    settings = settings.model_copy(update={"index_dir": args.index_dir})
    pipeline = build_pipeline(settings)

    rows = []
    for item in TEST_SET:
        result = pipeline.answer(item["query"])
        answerable = bool(item["gold_titles"])
        row = {
            "query": item["query"],
            "category": item["category"],
            "gold_titles": item["gold_titles"],
            "answer": result.answer,
            "retrieved_titles": result.retrieved_titles,
            "citations": result.citations,
            "refused": is_refusal(result.answer),
            "error": result.error,
            "timings_ms": result.timings_ms,
        }
        if answerable:
            row["expected_title_cited"] = any(f"[{title}]" in result.answer for title in item["gold_titles"])
        rows.append(row)
        status = "ERROR" if result.error else ("refused" if row["refused"] else "answered")
        print(f"[{item['category']:<16}] {status:<8} {item['query']}\n    {result.answer}")

    unanswerable_rows = [r for r in rows if not r["gold_titles"]]
    answerable_rows = [r for r in rows if r["gold_titles"]]
    summary = {
        "provider": settings.llm_provider,
        "model": settings.groq_model if settings.llm_provider == "groq" else settings.llm_provider,
        "unanswerable_refused": sum(r["refused"] for r in unanswerable_rows),
        "unanswerable_total": len(unanswerable_rows),
        "answerable_expected_title_cited": sum(r["expected_title_cited"] for r in answerable_rows),
        "answerable_refused": sum(r["refused"] for r in answerable_rows),
        "answerable_total": len(answerable_rows),
        "errors": sum(1 for r in rows if r["error"]),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    print(
        f"\nRefusals on unanswerable: {summary['unanswerable_refused']}/{summary['unanswerable_total']}\n"
        f"Expected title cited on answerable: "
        f"{summary['answerable_expected_title_cited']}/{summary['answerable_total']}\n"
        f"Errors: {summary['errors']}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"generation_{settings.llm_provider}_{stamp}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str) + "\n")
    print(f"Saved {out_path}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
