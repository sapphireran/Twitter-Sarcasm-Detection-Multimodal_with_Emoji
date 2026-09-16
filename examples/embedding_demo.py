#!/usr/bin/env python3
"""Mean-pool a tweet against tiny word and emoji tables.

This is the algorithm in ``AverageVectorPerTweet`` /
``AverageVectorPerEmoji`` / ``ml_read_data``, just with 8-d toy vectors
instead of GloVe Twitter 200-d and emoji2vec 200-d.

Run from the repository root:

    python examples/embedding_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.embeddings import average_vector_per_sequence, multimodal_concat
from examples.fixtures.tiny_tables import DIM, EMOJI_TABLE, WORD_TABLE
from examples.tokenize import tokenize_tweet

TWEETS = [
    "I loovee when people text back ... 😒 #sarcastictweet",
    "<user> Rest in peace & love to you and your family",
    "feeling like a million bucks after that chem 2 test . 😅 #not",
]


def format_vec(vec: np.ndarray) -> str:
    return " ".join(f"{value:+.2f}" for value in vec)


def main() -> int:
    names = [
        "pos-word",
        "neg-word",
        "sarc-tag",
        "grind",
        "affection",
        "sad-emoji",
        "deadpan-emoji",
        "happy-emoji",
    ]
    print("axis: " + "  ".join(f"{i}:{name}" for i, name in enumerate(names)))
    print()
    for tweet in TWEETS:
        tokens = tokenize_tweet(tweet)
        word = average_vector_per_sequence(tokens, WORD_TABLE, dim=DIM)
        emoji = average_vector_per_sequence(tokens, EMOJI_TABLE, dim=DIM)
        both = multimodal_concat(word[None, :], emoji[None, :])[0]
        print(tweet)
        print(f"  tokens          {tokens}")
        print(f"  word  (8)       {format_vec(word)}")
        print(f"  emoji (8)       {format_vec(emoji)}")
        print(f"  concat (16)     {format_vec(both)}")
        print()
    print(
        "Single-modal baselines in the notebooks use only the word mean. "
        "Multi-modal baselines concatenate the two means."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
