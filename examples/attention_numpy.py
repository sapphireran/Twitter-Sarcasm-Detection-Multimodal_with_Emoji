#!/usr/bin/env python3
"""Replay the course attention layer in NumPy on a toy hidden-state sequence.

``attention_layer.py`` implements Raffel-style attention on top of a
bidirectional LSTM. This script:

* builds a 6-step dummy sequence whose last two steps are scaled up
* scores it with the same ``tanh(x W + b)`` + softmax recipe
* optionally masks the final step to show renormalization
* walks the same math over a real tokenized tweet using random features

Usage:

    python3 examples/attention_numpy.py
    python3 examples/attention_numpy.py --tweet "I love 8am lectures #not 😒"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.attention import attention_forward, demo_sequence
from examples.lib.tokenize import tokenize_tweet


def _bar(weight: float, width: int = 20) -> str:
    filled = int(round(weight * width))
    return "#" * filled + "." * (width - filled)


def show_output(title: str, tokens: list[str], weights: np.ndarray, context: np.ndarray) -> None:
    print(f"## {title}")
    print(f"  context (first 6 dims): {np.array2string(context[0, :6], precision=3)}")
    print("  step  token            weight")
    for i, (token, weight) in enumerate(zip(tokens, weights[0])):
        print(f"  {i:4d}  {token:<16} {weight:6.3f}  {_bar(float(weight))}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tweet",
        default="I love walking to school at 6am #not 😒",
        help="Tweet whose tokens get random hidden states.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    demo = demo_sequence(seed=args.seed)
    tokens = [f"h{i}" for i in range(demo["x"].shape[0])]
    out = attention_forward(demo["x"], demo["W"], b=demo["b"])
    show_output("toy sequence (last two steps scaled up)", tokens, out.weights, out.context)

    mask = np.ones(demo["x"].shape[0], dtype=np.float64)
    mask[-1] = 0.0
    masked = attention_forward(demo["x"], demo["W"], b=demo["b"], mask=mask)
    show_output("same sequence with the last step masked", tokens, masked.weights, masked.context)

    tweet_tokens = tokenize_tweet(args.tweet)
    if not tweet_tokens:
        print("tweet produced no tokens", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    hidden = rng.normal(scale=0.2, size=(len(tweet_tokens), 8))
    # Nudge features on sarcasm markers so the weight bar is readable.
    for i, token in enumerate(tweet_tokens):
        if token in {"#not", "😒", "#sarcasm", "#sarcastictweet"} or token.startswith("#not"):
            hidden[i] += 1.6
    W = np.ones(8) / np.sqrt(8)
    tweet_out = attention_forward(hidden, W, b=np.zeros(len(tweet_tokens)))
    show_output(f"tweet: {args.tweet}", tweet_tokens, tweet_out.weights, tweet_out.context)
    print(
        "The Keras layer in attention_layer.py does this over 512-d BiLSTM "
        "states (2 * 256). The numbers here are pedagogical, not the trained weights."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
