#!/usr/bin/env python3
"""Replay AverageVectorPerTweet / concat fusion on a toy embedding table.

The real notebooks load 200-d Twitter GloVe. This script uses an 8-d handmade
table so you can see why bag-of-embeddings is permutation-invariant and why
concatenating an emoji mean is a different fusion story from putting emoji in
the sequence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.embeddings import average_vectors, cosine, embed_tweet, toy_table  # noqa: E402
from lib.tokenize import is_emoji_token, tokenize_tweet  # noqa: E402

PAIRS = [
    (
        "I just love having grungy ass hair 😑 #not",
        "I just love having grungy ass hair",
    ),
    (
        "3 more days until I'm reunited with my friends yay 😍",
        "I love waiting in line #not",
    ),
]


def _fmt(vec) -> str:
    return "[" + ", ".join(f"{x:+.2f}" for x in vec) + "]"


def show(text: str, table, dim: int) -> None:
    tokens = tokenize_tweet(text)
    word_only = {k: v for k, v in table.items() if not is_emoji_token(k)}
    emoji_only = {k: v for k, v in table.items() if is_emoji_token(k)}
    single = average_vectors(tokens, word_only, dim)
    multi = embed_tweet(text, word_only, dim, emoji_table=emoji_only)
    print(f"tweet:   {text}")
    print(f"tokens:  {tokens}")
    print(f"text-only mean ({dim}d): {_fmt(single)}")
    print(f"concat mean ({2 * dim}d): {_fmt(multi)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="optional extra tweet to embed")
    args = parser.parse_args()

    dim = 8
    table = toy_table(dim)
    print("Toy table directions (first 4 dims are polarity / situation / leak)\n")
    for key in ("love", "hate", "wait", "#not", "😑", "😍"):
        print(f"  {key:<8} {_fmt(table[key])}")
    print()

    for left, right in PAIRS:
        show(left, table, dim)
        show(right, table, dim)
        a = embed_tweet(left, table, dim)
        b = embed_tweet(right, table, dim)
        print(f"cosine({left!r}\n       vs {right!r})\n       = {cosine(a, b):.3f}\n")
        print("-" * 72)

    # Permutation invariance: same tokens, different order.
    s1 = "love waiting #not"
    s2 = "#not waiting love"
    c = cosine(embed_tweet(s1, table, dim), embed_tweet(s2, table, dim))
    print(f"permutation check: cosine({s1!r}, {s2!r}) = {c:.3f}  (should be 1.0)")

    if args.text:
        print()
        show(args.text, table, dim)


if __name__ == "__main__":
    main()
