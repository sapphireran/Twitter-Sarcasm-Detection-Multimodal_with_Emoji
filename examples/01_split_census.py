#!/usr/bin/env python3
"""Print split sizes, overlaps, and token-length stats."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccs2lab.report import markdown_table
from ccs2lab.splits import integrity, load_bundle
from ccs2lab.tokenize import tokenize_tweet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    bundle = load_bundle()
    report = integrity(bundle)

    rows = []
    for split in bundle.all_splits():
        lengths = [len(tokenize_tweet(text)) for text in split.texts]
        rows.append(
            (
                split.name,
                len(split),
                split.n_sarcastic,
                f"{split.n_sarcastic / len(split):.3f}",
                split.n_sincere,
                f"{statistics.mean(lengths):.2f}",
                statistics.median(lengths),
                max(lengths),
                min(lengths),
            )
        )

    headers = (
        "split",
        "n",
        "sarcastic",
        "sarc_rate",
        "sincere",
        "mean_tok",
        "median_tok",
        "max_tok",
        "min_tok",
    )
    print(markdown_table(headers, rows))
    print()
    print(f"train ∩ test (unique strings): {report.train_test_overlap}")
    print(f"test ∩ subtest: {report.test_subtest_overlap}")
    print(f"train ∩ subtest: {report.train_subtest_overlap}")
    print(f"subtest ⊆ test: {report.subtest_subset_of_test}")
    print(
        "expected sizes "
        f"train={report.expected_sizes['train']} "
        f"test={report.expected_sizes['test']} "
        f"subtest={report.expected_sizes['subtest']}"
    )
    if args.markdown:
        print("\n<!-- markdown table already printed -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
