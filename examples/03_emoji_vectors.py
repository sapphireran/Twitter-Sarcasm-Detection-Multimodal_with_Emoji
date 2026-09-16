#!/usr/bin/env python3
"""Inspect emoji2vec_twitter.bin without Gensim."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.io import load_split, repo_root
from examples.lib.tokenize import find_emoji
from examples.lib.word2vec_bin import cosine, cosine_neighbours, load_word2vec_binary, mean_in_vocab

PROBE = ("😂", "😒", "😍", "❤", "😭", "😊")


def main() -> None:
    path = repo_root() / "emoji2vec_twitter.bin"
    table = load_word2vec_binary(path)
    print(f"loaded {path.name}: {len(table)} tokens × {table.dim} dimensions")
    print()

    print("probe glyphs")
    for glyph in PROBE:
        if glyph not in table:
            print(f"  {glyph}  missing from the table")
            continue
        vec = table[glyph]
        neighbours = cosine_neighbours(table, glyph, k=5)
        neighbour_s = ", ".join(f"{tok} ({sim:.3f})" for tok, sim in neighbours)
        print(f"  {glyph}  L2={float((vec ** 2).sum() ** 0.5):.3f}  nearest: {neighbour_s}")

    print("\nmean emoji vector for a few test tweets")
    rows = [(text, label) for text, label in load_split("test") if find_emoji(text)]
    shown = 0
    means: list[tuple[str, int, object]] = []
    for text, label in rows:
        glyphs = find_emoji(text)
        mean = mean_in_vocab(table, glyphs)
        if mean is None:
            continue
        mark = "sarcastic" if label == 1 else "sincere"
        short = " ".join(text.split())
        if len(short) > 90:
            short = short[:89] + "…"
        print(f"  [{label} {mark}] glyphs={''.join(glyphs)}  {short}")
        means.append((short, label, mean))
        shown += 1
        if shown >= 4:
            break

    if len(means) >= 2:
        print("\ncosine between those tweet-level emoji means")
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                sim = cosine(means[i][2], means[j][2])
                print(f"  tweet {i} vs {j}: {sim:+.3f}")

    print("\nThis file is the 200-d table aligned with GloVe Twitter.")
    print("The 300-d original dump is emoji2vec.bin (same 1,661 tokens).")


if __name__ == "__main__":
    main()
