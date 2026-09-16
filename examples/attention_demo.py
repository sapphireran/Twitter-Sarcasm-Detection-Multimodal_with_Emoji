#!/usr/bin/env python3
"""Show Raffel-style temporal attention on a tiny synthetic batch.

The NumPy path matches ``attention_layer.Attention``: shared score vector,
optional per-timestep bias, tanh, masked softmax, weighted sum.

Usage:
    python3 examples/attention_demo.py
    python3 examples/attention_demo.py --steps 6 --features 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from sarcasm_lib.attention import masked_temporal_attention


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    print(name)
    for row in np.atleast_2d(matrix):
        cells = " ".join(f"{value:7.3f}" for value in row)
        print(f"  {cells}")


def build_demo_batch(steps: int, features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two tweets: one with a late 'punchline' feature, one fully padded after t=2."""
    x = np.zeros((2, steps, features), dtype=np.float64)
    # Row 0: content feature on early steps, sarcasm-ish feature on the last step.
    for t in range(steps):
        x[0, t, 0] = 1.0
    x[0, steps - 1, 1] = 4.0
    # Row 1: only the first two steps are real; the rest will be masked.
    x[1, 0, 0] = 1.0
    x[1, 1, 1] = 3.0
    mask = np.ones((2, steps), dtype=np.float64)
    mask[1, 2:] = 0.0
    weights = np.zeros(features, dtype=np.float64)
    weights[1] = 1.5  # attend to the sarcasm-ish channel
    return x, weights, mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--features", type=int, default=3)
    args = parser.parse_args(argv)
    if args.steps < 3:
        parser.error("--steps must be >= 3 so the mask demo has padding")

    x, weights, mask = build_demo_batch(args.steps, args.features)
    unmasked = masked_temporal_attention(x, weights)
    masked = masked_temporal_attention(x, weights, mask=mask)

    print("# Raffel temporal attention demo")
    print(
        "Channel 1 is the 'cue' dimension. W is zeros except W[1]=1.5, "
        "so softmax mass should pile on timesteps where channel 1 is large."
    )
    print()
    print(f"x shape        {x.shape}")
    print(f"W              {weights}")
    print(f"mask[1]        {mask[1]}")
    print()
    _print_matrix("attention weights without mask", unmasked.weights)
    _print_matrix("attention weights with mask", masked.weights)
    print()
    print("context without mask:", np.round(unmasked.context, 3))
    print("context with mask:   ", np.round(masked.context, 3))
    print()
    print(
        "row 0: last-step weight "
        f"unmasked={unmasked.weights[0, -1]:.3f}  "
        f"masked={masked.weights[0, -1]:.3f}"
    )
    print(
        "row 1: pad mass (steps >= 2) "
        f"unmasked={unmasked.weights[1, 2:].sum():.3f}  "
        f"masked={masked.weights[1, 2:].sum():.3f}"
    )
    print(
        "row 1: remaining weights sum to "
        f"{masked.weights[1].sum():.6f} (should be ~1)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
