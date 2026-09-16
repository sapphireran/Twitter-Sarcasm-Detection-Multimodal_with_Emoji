#!/usr/bin/env python3
"""Walk a toy sequence through the Raffel attention used by the BiLSTM.

The Keras layer in ``attention_layer.py`` is a weighted sum of encoder
states. This script uses readable token embeddings so you can see the
weights move toward the sarcasm cue.

Run from the repository root:

    python examples/attention_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.attention import attention_forward, uniform_attention
from examples.embeddings import lookup
from examples.fixtures.tiny_tables import DIM, WORD_TABLE
from examples.tokenize import tokenize_tweet


def embed_tokens(tokens: list[str]) -> np.ndarray:
    rows = []
    for token in tokens:
        vec = lookup(token, WORD_TABLE)
        rows.append(np.zeros(DIM) if vec is None else vec)
    return np.stack(rows, axis=0)


def main() -> int:
    tweet = "I love walking to school #not"
    tokens = tokenize_tweet(tweet)
    sequence = embed_tokens(tokens)[None, ...]

    # A weight vector that looks for the sarcasm-hashtag axis (index 2).
    weight = np.zeros(DIM)
    weight[2] = 3.0
    weight[0] = 0.2

    context, attn = attention_forward(sequence, weight)
    mean = uniform_attention(sequence)

    print(f"tweet:   {tweet}")
    print(f"tokens:  {tokens}")
    print()
    print(f"{'token':<16} {'attn':>8} {'uniform':>8}")
    uniform = np.full(len(tokens), 1.0 / len(tokens))
    for token, a, u in zip(tokens, attn[0], uniform):
        print(f"{token:<16} {a:8.3f} {u:8.3f}")
    print()
    print("context (attention pool):", np.round(context[0], 3))
    print("context (mean pool):     ", np.round(mean[0], 3))
    print()
    print(
        "The attention weight on `#not` should dominate because W is aligned "
        "with the sarcasm-hashtag axis. Mean pooling treats every token equally."
    )

    # Padding mask: pretend the last two tokens are pad.
    mask = np.ones((1, len(tokens)))
    mask[0, -2:] = 0.0
    _, masked = attention_forward(sequence, weight, mask=mask)
    print()
    print("same scores, last two steps masked:")
    for token, a in zip(tokens, masked[0]):
        print(f"  {token:<16} {a:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
