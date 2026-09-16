#!/usr/bin/env python3
"""Toy GloVe + emoji2vec mean-pool, matching data_utils.ml_read_data fusion.

A real run uses 200-d KeyedVectors. Here the tables are 8-d and hand-built so
the concat story is visible without a gigabyte download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.mean_pool import average_rows, concat_modalities, cosine
from examples.lib.tweet_tokenize import tokenize_tweet

DIM = 8

# Axes (illustrative, not from GloVe):
# 0 polarity+, 1 polarity-, 2 work/school, 3 night, 4 body, 5 family,
# 6 intensity, 7 "web / deixis"
GLOVE: dict[str, list[float]] = {
    "i": [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.4],
    "love": [0.9, 0.0, 0.0, 0.0, 0.1, 0.2, 0.6, 0.0],
    "loovee": [0.8, 0.0, 0.0, 0.0, 0.1, 0.1, 0.9, 0.0],
    "when": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.3],
    "people": [0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.2],
    "text": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
    "back": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    "getting": [0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.2, 0.1],
    "home": [0.4, 0.0, 0.0, 0.2, 0.0, 0.5, 0.1, 0.0],
    "from": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "work": [0.0, 0.2, 0.9, 0.1, 0.0, 0.0, 0.2, 0.0],
    "at": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2],
    "3am": [0.0, 0.2, 0.1, 0.9, 0.0, 0.0, 0.5, 0.0],
    "and": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    "my": [0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3],
    "house": [0.2, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
    "being": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    "dirty": [0.0, 0.7, 0.0, 0.0, 0.1, 0.1, 0.4, 0.0],
    "#not": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.9, 0.3],
    "#sarcastictweet": [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.8, 0.5],
    "great": [0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0],
    "failed": [0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.6, 0.0],
    "both": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    "physics": [0.0, 0.1, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
    "exams": [0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.3, 0.0],
    "christmas": [0.6, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0],
    "days": [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1],
    "until": [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2],
    "sorry": [0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0],
    "you": [0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.4],
}

# emoji2vec-style: faces sit near the affect they usually mark.
EMOJI2VEC: dict[str, list[float]] = {
    "😒": [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0],
    "😑": [0.0, 0.7, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0],
    "😅": [0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
    "😭": [0.1, 0.8, 0.0, 0.0, 0.0, 0.3, 0.8, 0.0],
    "🌲": [0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.0],
    "😃": [0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.0],
    "🔫": [0.0, 0.6, 0.0, 0.0, 0.2, 0.0, 0.5, 0.0],
}


TWEETS = [
    ("sarc", "I loovee when people text back ... 😒 #sarcastictweet"),
    ("sarc", "Oh how I love getting home from work at 3am and my house being dirty #not"),
    ("sarc", "100% failed both physics exams great"),
    ("nons", "100 days until Christmas! 🌲"),
    ("nons", "sorry you failed both physics exams"),
]


def _table(raw: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    glove = _table(GLOVE)
    e2v = _table(EMOJI2VEC)
    token_lists = [tokenize_tweet(text) for _, text in TWEETS]
    word = average_rows(token_lists, glove, DIM)
    emoji = average_rows(token_lists, e2v, DIM)
    fused = concat_modalities(word, emoji)

    print("dim(word) =", word.shape, "dim(fused) =", fused.shape)
    print()
    for i, (label, text) in enumerate(TWEETS):
        oov_w = [t for t in token_lists[i] if t not in glove]
        hit_e = [t for t in token_lists[i] if t in e2v]
        print(f"[{label}] {text}")
        print(f"  tokens     {token_lists[i]}")
        print(f"  word OOV   {oov_w or '—'}")
        print(f"  emoji hits {hit_e or '—'}")
        print(f"  word[0:4]  {np.round(word[i, :4], 3)}")
        print(f"  emoji[0:4] {np.round(emoji[i, :4], 3)}")
        print()

    print("cosine(word-only) between tweet 0 (sarc+face) and others")
    for i in range(1, len(TWEETS)):
        print(f"  vs {i} ({TWEETS[i][0]}): {cosine(word[0], word[i]):.3f}")
    print("cosine(fused) between tweet 0 and others")
    for i in range(1, len(TWEETS)):
        print(f"  vs {i} ({TWEETS[i][0]}): {cosine(fused[0], fused[i]):.3f}")
    print()
    print(
        "The empty emoji half of tweet 2 / 4 is a zero block, same as "
        "AverageVectorPerEmoji on a text-only line. Concat still doubles "
        "the dimension; that is why SVM can lose a bit on the full test set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
