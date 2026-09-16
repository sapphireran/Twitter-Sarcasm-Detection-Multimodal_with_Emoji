#!/usr/bin/env python3
"""Walk through Raffel attention on a three-token toy tweet.

The Keras layer in attention_layer.py scores each time step with
tanh(x_t · W + b_t), softmaxes those scores, and returns the weighted sum.
This script uses the NumPy twin in sarcasm_lib.attention so you can see the
α vector without loading TensorFlow or the 2.5M-parameter checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarcasm_lib.attention import explain_attention
from sarcasm_lib.tokenize import tokenize_tweet


TOY_TWEET = "I love Monday mornings #not"
# Two-dimensional hand-built embeddings: dim 0 ~ "positive surface", dim 1 ~
# "explicit sarcasm cue". #not is the only token with a large cue component.
TOY_ROWS = {
    "i": np.array([0.15, 0.00]),
    "love": np.array([0.95, 0.05]),
    "monday": np.array([0.20, 0.00]),
    "mornings": np.array([0.25, 0.00]),
    "#not": np.array([0.10, 1.20]),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweet", default=TOY_TWEET)
    parser.add_argument(
        "--cue-weight",
        type=float,
        default=1.4,
        help="W[1]: how hard the layer looks for the sarcasm-cue dimension",
    )
    return parser


def embed(tokens: list[str]) -> np.ndarray:
    rows = [TOY_ROWS.get(token, np.array([0.05, 0.00])) for token in tokens]
    return np.stack(rows, axis=0)[None, ...]


def main() -> None:
    args = build_parser().parse_args()
    tokens = tokenize_tweet(args.tweet)
    if not tokens:
        raise SystemExit("tweet produced no tokens")

    sequence = embed(tokens)
    weights = np.array([0.2, args.cue_weight])
    walk = explain_attention(sequence, weights)

    print(f"tweet:   {args.tweet}")
    print(f"tokens:  {tokens}")
    print(f"W:       {weights.tolist()}")
    print()
    print(f"{'idx':>4} {'token':<12} {'score':>8} {'alpha':>8}  bar")
    for index, token in enumerate(tokens):
        score = walk.scores[0, index]
        alpha = walk.alphas[0, index]
        bar = "#" * int(round(alpha * 40))
        print(f"{index:4d} {token:<12} {score:8.3f} {alpha:8.3f}  {bar}")

    print()
    print(f"context vector c = Σ α_t x_t: {np.round(walk.context[0], 3).tolist()}")
    top = walk.top_steps(k=min(3, len(tokens)))
    pretty = ", ".join(f"{tokens[i]}={alpha:.3f}" for i, alpha in top)
    print(f"top steps: {pretty}")
    print()
    print(
        "The Keras layer uses the same pooling after two Bi-LSTM layers. "
        "If the network has aligned the second feature with #not / polarity "
        "clash, attention puts most of its mass on that step — the same way "
        "this toy W does."
    )


if __name__ == "__main__":
    main()
