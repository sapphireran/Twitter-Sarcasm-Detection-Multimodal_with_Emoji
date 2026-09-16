#!/usr/bin/env python3
"""Emoji co-occurrence vs. sarcasm label.

The multimodal branch of this project averages emoji2vec rows that fire on a
tweet. Before looking at embeddings, this script asks a simpler question: which
pictographs actually co-occur with the sarcastic class?

Run::

    python3 examples/02_emoji_cooccurrence.py
    python3 examples/02_emoji_cooccurrence.py --split subtest --top 15
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.emoji import emoji_code_points
from examples.common.io import load_split


def _pmi(joint: int, n_emoji: int, n_class: int, n: int) -> float:
    """Pointwise mutual information between an emoji and a class."""
    if joint == 0 or n_emoji == 0 or n_class == 0:
        return float("-inf")
    p_e = n_emoji / n
    p_c = n_class / n
    p_ec = joint / n
    return math.log(p_ec / (p_e * p_c))


def collect(split):
    emoji_total = Counter()
    emoji_by_label = {0: Counter(), 1: Counter()}
    tweets_with_emoji = {0: 0, 1: 0}
    for text, label in split.pairs():
        emojis = emoji_code_points(text)
        if emojis:
            tweets_with_emoji[label] += 1
        for e in set(emojis):  # document frequency, not raw count
            emoji_total[e] += 1
            emoji_by_label[label][e] += 1
    return emoji_total, emoji_by_label, tweets_with_emoji


def table(split, top: int) -> str:
    emoji_total, by_label, tweets_with = collect(split)
    n = len(split)
    n1 = split.n_sarcastic
    n0 = split.n_literal
    rows = []
    for e, df in emoji_total.most_common():
        c0 = by_label[0][e]
        c1 = by_label[1][e]
        pmi1 = _pmi(c1, df, n1, n)
        pmi0 = _pmi(c0, df, n0, n)
        sarc_rate = c1 / df if df else 0.0
        rows.append((e, df, c0, c1, sarc_rate, pmi1, pmi0))
    lines = [
        f"split={split.name}  n={n}  literal={n0}  sarcastic={n1}",
        f"tweets with ≥1 emoji: literal={tweets_with[0]}  sarcastic={tweets_with[1]}",
        "",
        f"{'emoji':<8} {'df':>6} {'lit':>6} {'sarc':>6} {'sarc%':>8} {'PMI_s':>8} {'PMI_l':>8}",
        "-" * 60,
    ]
    for e, df, c0, c1, rate, pmi1, pmi0 in rows[:top]:
        lines.append(
            f"{e:<8} {df:6d} {c0:6d} {c1:6d} {100*rate:8.1f} {pmi1:8.3f} {pmi0:8.3f}"
        )
    lines.append("")
    lines.append("Highest PMI with sarcastic (min df=20 on train, 3 otherwise):")
    min_df = 20 if split.name == "train" else 3
    ranked = [r for r in rows if r[1] >= min_df]
    ranked.sort(key=lambda r: r[5], reverse=True)
    for e, df, c0, c1, rate, pmi1, pmi0 in ranked[:10]:
        lines.append(
            f"  {e}  df={df:4d}  sarc_rate={100*rate:5.1f}%  PMI={pmi1:.3f}"
        )
    lines.append("")
    lines.append("Highest PMI with literal:")
    ranked.sort(key=lambda r: r[6], reverse=True)
    for e, df, c0, c1, rate, pmi1, pmi0 in ranked[:10]:
        lines.append(
            f"  {e}  df={df:4d}  sarc_rate={100*rate:5.1f}%  PMI_lit={pmi0:.3f}"
        )
    lines.append("")
    lines.append(
        "Reading: 😒 and 😑 often mark deadpan affect and lean sarcastic in this"
    )
    lines.append(
        "export; ❤ / 😍 lean literal. 😂 is the most frequent pictograph in both"
    )
    lines.append("classes, so it is a weak class cue on its own.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test", "subtest"])
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    split = load_split(args.split)
    print(table(split, args.top), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
