#!/usr/bin/env python3
"""Walk through the project's temporal attention on a toy 3-step sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.attention_numpy import attention_pool, uniform_attention_pool


def _demo_batch() -> tuple[np.ndarray, list[str]]:
    """One sarcastic-style sequence: setup, contrast, flip cue."""
    tokens = ["love", "dirty-house", "#not"]
    # 4-d toy embeddings: [positive, negative, hashtag-flip, emoji-deadpan]
    sequence = np.array(
        [
            [0.92, 0.05, 0.00, 0.10],
            [0.10, 0.88, 0.00, 0.35],
            [0.05, 0.10, 0.97, 0.20],
        ],
        dtype=np.float64,
    )
    return sequence[None, ...], tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sequences, tokens = _demo_batch()
    # Prefer the flip-cue and negative dimensions, similar to a trained scorer.
    weight = np.array([0.15, 0.45, 1.10, 0.35], dtype=np.float64)
    bias = np.array([0.00, 0.05, 0.10], dtype=np.float64)
    context, attn = attention_pool(sequences, weight, bias=bias)
    mean = uniform_attention_pool(sequences)

    print("Temporal attention on a 3-token sarcastic sketch")
    print("================================================")
    print()
    print("tokens:     ", tokens)
    print("embeddings: love=[+], dirty-house=[-], #not=[flip]")
    print()
    print("learned score direction:", np.round(weight, 3).tolist())
    print("attention weights:      ", np.round(attn[0], 3).tolist())
    print("attention context:      ", np.round(context[0], 3).tolist())
    print("mean-pool context:      ", np.round(mean[0], 3).tolist())
    print()
    print(
        "The #not step receives most of the mass, so the pooled vector keeps "
        "the flip cue instead of averaging it away. That is the same reason "
        "the course model puts Attention() on top of Bi-LSTM states."
    )
    print()

    rng = np.random.default_rng(args.seed)
    random_weight = rng.normal(0, 0.4, size=4)
    _, random_attn = attention_pool(sequences, random_weight)
    print("random-weight attention:", np.round(random_attn[0], 3).tolist())
    print("A badly initialized scorer can attend to the setup token instead.")


if __name__ == "__main__":
    main()
