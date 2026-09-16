#!/usr/bin/env python3
"""Show Raffel attention putting mass on one hidden step.

The numbers are synthetic — there is no trained LSTM here — but the formula is
the same one as ``attention_layer.Attention.call``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.attention_numpy import AttentionWeights, attention_forward, peaked_weights


def main() -> int:
    rng = np.random.default_rng(7)
    steps, features = 8, 6
    # Hidden states: mostly noise, with step 5 (0-based) clearly larger.
    hidden = rng.normal(0.0, 0.15, size=(steps, features))
    hidden[5] = hidden[5] + 1.4

    weights = peaked_weights(steps, features, peak=5)
    context, alphas = attention_forward(hidden, weights)

    print("=== Raffel attention (numpy port of attention_layer.py) ===")
    print(f"hidden shape {hidden.shape}  context shape {context.shape}")
    print("alphas:")
    for t, alpha in enumerate(alphas):
        bar = "#" * int(round(alpha * 40))
        marker = "  <-- peak" if t == 5 else ""
        print(f"  t={t}  {alpha:6.3f}  {bar}{marker}")
    print(f"sum(alphas) = {alphas.sum():.6f}  (should be ~1)")
    print(f"argmax      = {int(alphas.argmax())}  (should be 5)")

    # Mask the peak away and the mass should move.
    mask = np.ones((steps,), dtype=np.float64)
    mask[5] = 0.0
    _context2, alphas2 = attention_forward(hidden, weights, mask=mask)
    print()
    print("=== same weights, step 5 masked as pad ===")
    print("alphas:")
    for t, alpha in enumerate(alphas2):
        bar = "#" * int(round(alpha * 40))
        marker = "  (masked)" if t == 5 else ""
        print(f"  t={t}  {alpha:6.3f}  {bar}{marker}")
    print(f"masked alpha[5] = {alphas2[5]:.3e}  (should be ~0)")

    # Bias-free, uniform-ish hidden → near-uniform attention.
    flat = np.ones((4, 3), dtype=np.float64)
    ctx, uni = attention_forward(flat, AttentionWeights(W=np.ones(3), b=None))
    print()
    print("=== flat hidden, no bias → near-uniform alphas ===")
    print("alphas", np.round(uni, 3), " context", np.round(ctx, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
