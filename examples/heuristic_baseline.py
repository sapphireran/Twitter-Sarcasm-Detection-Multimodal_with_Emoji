#!/usr/bin/env python3
"""Evaluate the explicit-cue heuristic on every official split.

This is the number that belongs next to the Bi-LSTM table: how far you get if
you only trust #not / #sarcastictweet and a positive-opener + negative-emoji
clash. Precision on test and subtest is 1.0; recall is the gap the network
still has to close.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarcasm_lib.heuristic import predict_many, score_tweet
from sarcasm_lib.io import load_all_splits
from sarcasm_lib.metrics import binary_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--errors",
        type=int,
        default=5,
        help="false negatives to print from the test split",
    )
    return parser


def metrics_dict(name: str, labels: list[int], preds: list[int]) -> dict[str, object]:
    result = binary_metrics(labels, preds)
    return {
        "split": name,
        "accuracy": round(result.accuracy, 4),
        "precision": round(result.precision, 4),
        "recall": round(result.recall, 4),
        "f1": round(result.f1, 4),
        "tp": result.counts.true_positive,
        "fp": result.counts.false_positive,
        "tn": result.counts.true_negative,
        "fn": result.counts.false_negative,
    }


def main() -> None:
    args = build_parser().parse_args()
    splits = load_all_splits()
    rows = []
    for name, split in splits.items():
        preds = predict_many(split.texts, threshold=args.threshold)
        rows.append(metrics_dict(name, split.labels, preds))

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"Explicit-cue heuristic (threshold={args.threshold})")
    print(f"{'split':<8} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'tp':>6} {'fp':>5} {'fn':>5}")
    for row in rows:
        print(
            f"{row['split']:<8} {row['accuracy']:7.4f} {row['precision']:7.4f} "
            f"{row['recall']:7.4f} {row['f1']:7.4f} {row['tp']:6d} {row['fp']:5d} {row['fn']:5d}"
        )

    print()
    print(f"Sarcastic test tweets the heuristic missed (up to {args.errors}):")
    test = splits["test"]
    shown = 0
    for text, label in test.pairs():
        result = score_tweet(text, threshold=args.threshold)
        if label == 1 and result.predicted == 0:
            snippet = " ".join(text.split())
            print(f"  score={result.score:.2f}  {snippet[:180]}")
            shown += 1
            if shown >= args.errors:
                break


if __name__ == "__main__":
    main()
