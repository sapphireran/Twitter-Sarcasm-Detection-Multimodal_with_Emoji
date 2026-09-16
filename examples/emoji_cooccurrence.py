#!/usr/bin/env python3
"""Emoji × label co-occurrence on the local CSVs.

This is a contingency table, not a causal claim. An emoji with a high sarcastic
rate may just travel with #not. Use --strip-leak to recompute after dropping
tweets that contain the distant-supervision hashtags.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.features import has_leak_hashtag  # noqa: E402
from lib.io import load_split  # noqa: E402
from lib.tokenize import tokenize_tweet  # noqa: E402


def _is_emoji(tok: str) -> bool:
    return len(tok) == 1 and ord(tok) > 255


def table(split_name: str, top: int, min_count: int, strip_leak: bool) -> None:
    split = load_split(split_name)
    stats = defaultdict(lambda: {"n": 0, "pos": 0})
    used = 0
    for text, label in zip(split.texts, split.labels):
        if strip_leak and has_leak_hashtag(text):
            continue
        used += 1
        seen = {tok for tok in tokenize_tweet(text) if _is_emoji(tok)}
        for emo in seen:
            stats[emo]["n"] += 1
            stats[emo]["pos"] += int(label == 1)

    ranked = [
        (emo, rec["n"], rec["pos"] / rec["n"] if rec["n"] else 0.0)
        for emo, rec in stats.items()
        if rec["n"] >= min_count
    ]
    ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)

    print(f"split={split_name}  tweets_used={used}  strip_leak={strip_leak}  min_count={min_count}")
    print(f"{'emoji':<8} {'tweets':>8} {'sarcastic_rate':>16}")
    print("-" * 36)
    for emo, n, rate in ranked[:top]:
        print(f"{emo:<8} {n:8d} {rate:16.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "test", "subtest"))
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument(
        "--strip-leak",
        action="store_true",
        help="drop tweets containing #not / #sarcasm / #yeahright before counting",
    )
    args = parser.parse_args()
    table(args.split, args.top, args.min_count, args.strip_leak)


if __name__ == "__main__":
    main()
