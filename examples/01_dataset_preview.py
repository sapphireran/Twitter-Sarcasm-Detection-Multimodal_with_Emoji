#!/usr/bin/env python3
"""Print split sizes, class balance, and a few labelled rows."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.io import SPLITS, class_counts, load_split
from examples.lib.tokenize import find_emoji, find_hashtags


def _short(text: str, limit: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarise(name: str) -> None:
    rows = load_split(name)
    counts = class_counts(rows)
    n = len(rows)
    with_emoji = sum(1 for text, _ in rows if find_emoji(text))
    with_tag = sum(1 for text, _ in rows if find_hashtags(text))
    sarcastic = counts[1]
    print(f"## {name}")
    print(f"  rows          {n}")
    print(f"  sarcastic     {sarcastic} ({sarcastic / n:.1%})")
    print(f"  non-sarcastic {counts[0]} ({counts[0] / n:.1%})")
    print(f"  with emoji    {with_emoji} ({with_emoji / n:.1%})")
    print(f"  with hashtag  {with_tag} ({with_tag / n:.1%})")
    print("  examples:")
    shown = {0: 0, 1: 0}
    for text, label in rows:
        if shown[label] >= 2:
            continue
        mark = "sarcastic" if label == 1 else "sincere  "
        print(f"    [{label} {mark}] {_short(text)}")
        shown[label] += 1
        if shown[0] >= 2 and shown[1] >= 2:
            break
    print()


def main() -> None:
    print("Shipped CSV splits (headerless sentence/label pairs)\n")
    for name in SPLITS:
        summarise(name)
    print("See docs/dataset.md for hashtag leakage and emoji tables.")


if __name__ == "__main__":
    main()
