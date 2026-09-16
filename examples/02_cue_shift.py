#!/usr/bin/env python3
"""Measure hashtag / emoji shift across train, test, and subtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccs2lab.cues import collect_counts
from ccs2lab.metrics import odds_ratio, wilson_interval
from ccs2lab.report import markdown_table
from ccs2lab.splits import load_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cues",
        default="explicit,not_tag,sarcasm_tag,emoji,contrast",
        help="Comma-separated cue keys from collect_counts.",
    )
    args = parser.parse_args()
    wanted = [item.strip() for item in args.cues.split(",") if item.strip()]
    bundle = load_bundle()

    headers = (
        "split",
        "cue",
        "hits",
        "coverage",
        "cov_lo",
        "cov_hi",
        "hit_prec",
        "odds_ratio",
    )
    rows = []
    for split in bundle.all_splits():
        counts = collect_counts(split.texts, split.labels)
        for key in wanted:
            bucket = counts[key]
            _p, lo, hi = wilson_interval(bucket.hits, bucket.n)
            miss_pos = bucket.sarcastic - bucket.hits_sarcastic
            miss_neg = (bucket.n - bucket.sarcastic) - (bucket.hits - bucket.hits_sarcastic)
            ratio = odds_ratio(
                bucket.hits_sarcastic,
                bucket.hits - bucket.hits_sarcastic,
                miss_pos,
                miss_neg,
            )
            rows.append(
                (
                    split.name,
                    key,
                    bucket.hits,
                    bucket.coverage,
                    lo,
                    hi,
                    bucket.hit_precision,
                    ratio,
                )
            )
    print(markdown_table(headers, rows))
    print()
    print(
        "Wilson intervals are 95%. Odds ratios use a 0.5 Haldane–Anscombe "
        "correction. `#not` coverage jumping from train to test is the "
        "main leakage number."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
