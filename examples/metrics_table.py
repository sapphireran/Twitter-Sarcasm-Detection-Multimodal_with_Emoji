#!/usr/bin/env python3
"""Pretty-print the recorded 2023 notebook metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.recorded_metrics import CONDITIONS, markdown_table, rows_for_metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        default="accuracy",
        choices=("accuracy", "f1", "recall", "precision"),
    )
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    print(f"Recorded {args.metric} from get_metrics_of_models.ipynb")
    print("W = word / GloVe only. WE = word + emoji2vec.")
    print("subtest = emoji-bearing slice of the official test set.")
    print()
    if args.markdown:
        print(markdown_table(args.metric))
        return

    header = f"{'model':<22}" + "".join(f"{name:>22}" for name in CONDITIONS)
    print(header)
    for row in rows_for_metric(args.metric):
        cells = "".join(f"{value:>22}" for value in row.as_percent())
        print(f"{row.model:<22}{cells}")


if __name__ == "__main__":
    main()
