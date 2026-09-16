#!/usr/bin/env python3
"""Compare emoji inventories for sarcastic vs non-sarcastic tweets."""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.io_utils import load_split
from examples.lib.tweet_features import extract_emojis


def _smoothed_rate(count: int, total: int, alpha: float = 0.5) -> float:
    return (count + alpha) / (total + 2 * alpha)


def ranked_emoji(split_name: str, limit: int) -> list[tuple[str, int, int, float]]:
    split = load_split(split_name)
    sarcastic = collections.Counter()
    sincere = collections.Counter()
    for text, label in split.pairs():
        bag = extract_emojis(text)
        if label == 1:
            sarcastic.update(bag)
        else:
            sincere.update(bag)

    n_s = sum(1 for _label in split.labels if _label == 1)
    n_n = sum(1 for _label in split.labels if _label == 0)
    glyphs = set(sarcastic) | set(sincere)
    ranked = []
    for glyph in glyphs:
        pos = sarcastic[glyph]
        neg = sincere[glyph]
        log_odds = math.log(_smoothed_rate(pos, n_s) / _smoothed_rate(neg, n_n))
        ranked.append((glyph, pos, neg, log_odds))
    ranked.sort(key=lambda row: row[3], reverse=True)
    return ranked[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "test", "subtest"))
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    print(f"Emoji log-odds of sarcasm on the {args.split} split")
    print("positive log-odds => more common in sarcastic tweets")
    print()
    print(f"{'emoji':<8}{'y=1':>8}{'y=0':>8}{'log-odds':>12}")
    for glyph, pos, neg, log_odds in ranked_emoji(args.split, args.limit):
        print(f"{glyph:<8}{pos:8d}{neg:8d}{log_odds:12.3f}")
    print()
    print("On the official test set, 😒 😑 😅 and 🔫 lean sarcastic.")
    print("On train, 😂 and ❤ are frequent in both classes and weaker as flips.")


if __name__ == "__main__":
    main()
