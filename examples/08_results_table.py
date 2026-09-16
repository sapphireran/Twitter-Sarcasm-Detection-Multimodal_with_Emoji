#!/usr/bin/env python3
"""Pretty-print the 2023 notebook metrics checked into docs/metrics/.

Run::

    python3 examples/08_results_table.py
    python3 examples/08_results_table.py --metric f1
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "metrics" / "published_metrics.csv"


def load_rows():
    with METRICS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{100 * float(value):6.2f}"
    except ValueError:
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        default="all",
        choices=["all", "accuracy", "f1", "precision", "recall"],
    )
    args = parser.parse_args()
    rows = load_rows()
    wanted = {"accuracy", "f1", "precision", "recall"} if args.metric == "all" else {args.metric}

    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        if row["metric"] in wanted:
            grouped[row["metric"]].append(row)

    for metric in ("accuracy", "f1", "recall", "precision"):
        block = grouped.get(metric)
        if not block:
            continue
        print(f"{metric}")
        print("-" * len(metric))
        print(
            f"{'model':22} {'test W':>8} {'test WE':>8} {'sub W':>8} {'sub WE':>8}"
        )
        for row in block:
            print(
                f"{row['model']:22} {fmt(row['test_W']):>8} {fmt(row['test_WE']):>8} "
                f"{fmt(row['subtest_W']):>8} {fmt(row['subtest_WE']):>8}"
            )
        print()

    print(f"source file: {METRICS.relative_to(ROOT)}")
    print("These are transcribed notebook outputs, not a live re-evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
