#!/usr/bin/env python3
"""Print the recorded June 2023 accuracy / F1 / precision / recall tables.

These numbers are copied from the executed course notebooks, not recomputed.
The script exists so ``docs/results.md`` can stay in sync with a single
source of truth in ``examples/lib/metrics.py``.

Usage:

    python3 examples/report_results.py
    python3 examples/report_results.py --metric f1
    python3 examples/report_results.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.metrics import RECORDED_RESULTS, format_results_table, pct


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        default="accuracy",
        choices=("accuracy", "f1", "precision", "recall"),
    )
    parser.add_argument("--all", action="store_true", help="Print every metric table.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    print("Recorded course-project results")
    print(RECORDED_RESULTS["source"])
    print("W = GloVe / word only.  WE = GloVe concatenated with emoji2vec.\n")
    splits = RECORDED_RESULTS["splits"]
    print("Splits")
    for name, info in splits.items():
        print(
            f"  {name:<8} n={info['n']:<6} "
            f"sarcastic={info['sarcastic']:<6} non={info['non_sarcastic']}"
        )
    print()

    metrics = ("accuracy", "f1", "precision", "recall") if args.all else (args.metric,)
    for metric in metrics:
        print(f"### {metric}")
        print(format_results_table(metric))
        print()

    rule = RECORDED_RESULTS["rule_baseline_test"]
    print("### surface-cue rule on official test (recomputed from the CSVs)")
    print(
        f"| {rule['label']} | acc {pct(rule['accuracy'])} | "
        f"P {pct(rule['precision'])} | R {pct(rule['recall'])} | "
        f"F1 {pct(rule['f1'])} |"
    )
    print(
        f"Explicit sarcasm hashtags cover {pct(rule['explicit_marker_coverage'])} "
        "of sarcastic test tweets. High precision, incomplete recall."
    )
    print()
    print(
        "Takeaway: emoji2vec barely moves the official test set for trees / SVM, "
        "but it adds 2–5 points on the emoji-rich subtest, and the BiLSTM + "
        "attention model is the only one that clears 87% test accuracy."
    )
    if args.json:
        print()
        print(json.dumps(RECORDED_RESULTS, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
