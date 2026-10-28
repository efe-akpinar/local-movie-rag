"""Turn Locust's ``<prefix>_stats.csv`` into a small markdown table.

python scripts/summarize_loadtest.py results/loadtest/cloudrun_stats.csv [--out summary.md]
"""

import argparse
import csv
import sys
from pathlib import Path

COLUMNS = ["Requests", "Failures", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Req/s"]


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize(stats_csv: Path) -> str:
    with stats_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{stats_csv} has no rows")

    lines = [
        "| Endpoint | " + " | ".join(COLUMNS) + " |",
        "|---|" + "---:|" * len(COLUMNS),
    ]
    for row in rows:
        name = row["Name"] if row["Name"] == "Aggregated" else f"{row['Type']} {row['Name']}".strip()
        values = [
            f"{int(_number(row['Request Count'])):,}",
            f"{int(_number(row['Failure Count'])):,}",
            f"{_number(row['50%']):.0f}",
            f"{_number(row['95%']):.0f}",
            f"{_number(row['99%']):.0f}",
            f"{_number(row['Requests/s']):.2f}",
        ]
        if row["Name"] == "Aggregated":
            name, values = f"**{name}**", [f"**{v}**" for v in values]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stats_csv", type=Path)
    parser.add_argument("--out", type=Path, help="write the table to this file as well")
    args = parser.parse_args()

    table = summarize(args.stats_csv)
    print(table, end="")
    if args.out:
        args.out.write_text(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
