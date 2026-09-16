#!/usr/bin/env python3
"""Print class balance, emoji rate, and top hashtags for the real CSV splits.

    python3 examples/inspect_dataset.py
    python3 examples/inspect_dataset.py --split subtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python3 examples/inspect_dataset.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dataset_stats import SPLIT_FILES, summarize_all, summarize_split


def _fmt_rate(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _print_summary(summary) -> None:
    print(f"## {summary.name}")
    print(f"  rows              {summary.n}")
    print(f"  sarcastic / literal {summary.n_sarcastic} / {summary.n_literal}")
    print(f"  sarcasm rate      {_fmt_rate(summary.sarcasm_rate)}")
    print(f"  tweets with emoji {_fmt_rate(summary.emoji_rate)}")
    print(f"  tweets with #tag  {_fmt_rate(summary.hashtag_rate)}")
    print(f"  #not/#sarcasm…    {_fmt_rate(summary.sarcasm_marker_rate)}")
    print(
        f"  tokens            mean {summary.mean_tokens:.1f}  "
        f"p50 {summary.p50_tokens:.0f}  p90 {summary.p90_tokens:.0f}"
    )
    print("  top hashtags")
    if not summary.top_hashtags:
        print("    (none)")
        return
    for tag, count in summary.top_hashtags:
        print(f"    {count:5d}  {tag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", *SPLIT_FILES.keys()],
        help="which CSV pair to summarize",
    )
    args = parser.parse_args(argv)

    print("Personal CCS2 sarcasm dataset — local CSV inspection")
    print("No embeddings, no network, no sklearn.\n")

    summaries = summarize_all() if args.split == "all" else [summarize_split(args.split)]
    for summary in summaries:
        _print_summary(summary)
        print()

    if args.split == "all":
        train, test, sub = summaries
        print("## subtest vs train (why fusion shows up there)")
        print(
            f"  sarcasm rate   train {_fmt_rate(train.sarcasm_rate)} → "
            f"subtest {_fmt_rate(sub.sarcasm_rate)}"
        )
        print(
            f"  emoji rate     train {_fmt_rate(train.emoji_rate)} → "
            f"subtest {_fmt_rate(sub.emoji_rate)}"
        )
        print(
            f"  marker rate    train {_fmt_rate(train.sarcasm_marker_rate)} → "
            f"subtest {_fmt_rate(sub.sarcasm_marker_rate)}"
        )
        print(
            f"  test is balanced ({_fmt_rate(test.sarcasm_rate)} sarcastic) "
            "and is the headline number in docs/results.md"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
