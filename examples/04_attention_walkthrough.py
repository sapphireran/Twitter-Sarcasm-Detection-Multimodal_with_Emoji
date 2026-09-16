#!/usr/bin/env python3
"""Replay attention_layer.Attention.call on a tiny synthetic batch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.attention import temporal_attention


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}  shape={tuple(matrix.shape)}")
    for i, row in enumerate(matrix):
        cells = "  ".join(f"{value:+.3f}" for value in np.atleast_1d(row))
        print(f"  [{i}] {cells}")


def main() -> None:
    # Step 2 is a sarcasm-cue direction. The other steps are near-orthogonal filler.
    filler = np.array([1.0, 0.0, 0.0])
    cue = np.array([0.0, 1.0, 0.0])
    other = np.array([0.0, 0.0, 1.0])
    sequence = np.stack([filler, other, cue, filler], axis=0)
    x = np.stack([sequence, sequence], axis=0)  # batch of 2 identical tweets

    weight = np.array([0.1, 2.5, -0.2])  # aligned with the cue axis
    bias = np.array([0.0, 0.0, 0.15, 0.0])  # small extra preference for step 2
    mask = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],  # second row cannot see the cue
        ]
    )

    context, weights = temporal_attention(x, weight, bias=bias, mask=mask)

    print("Synthetic batch: 2 tweets × 4 steps × 3 features")
    print("  step 0 filler [1,0,0]")
    print("  step 1 other  [0,0,1]")
    print("  step 2 cue    [0,1,0]   (masked out on tweet 1)")
    print("  step 3 filler [1,0,0]")
    print("  W is pointed at the cue axis, matching Attention.call.\n")

    _print_matrix("softmax weights α", weights)
    print(f"  row sums: {weights.sum(axis=1)}")
    _print_matrix("pooled context c", context)

    print("\nWhat to look for")
    print("  • tweet 0 puts most mass on step 2; c[0] leans toward the cue axis")
    print("  • tweet 1 has α[:,2] = 0; the remaining weights re-normalise")
    print("  • both rows of α are non-negative and sum to 1")
    print("\nThis is pooling, not decoder attention. See docs/attention.md.")


if __name__ == "__main__":
    main()
