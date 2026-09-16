#!/usr/bin/env python3
"""NumPy replay of `attention_layer.Attention.call`.

The Keras layer is:

    e = tanh(x @ W + b)          # b is per-timestep
    a = softmax(e)               # mask after exp, + epsilon
    out = sum(a * x, axis=steps)

This script builds a 4-step toy sequence, prints the scores, and
checks that a fully-masked step gets ~zero mass.

    python examples/attention_demo.py
"""

from __future__ import annotations

import argparse

import numpy as np


def attention(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    epsilon: float = np.finfo(np.float32).eps,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (context, alpha, eij) with shapes (B, F), (B, T), (B, T)."""
    if x.ndim != 3:
        raise ValueError(f"expected (batch, steps, features), got {x.shape}")
    eij = np.squeeze(x @ w[:, None], axis=-1)
    if b is not None:
        eij = eij + b
    eij = np.tanh(eij)
    alpha = np.exp(eij)
    if mask is not None:
        alpha = alpha * mask.astype(alpha.dtype)
    alpha = alpha / (np.sum(alpha, axis=1, keepdims=True) + epsilon)
    context = np.sum(x * alpha[:, :, None], axis=1)
    return context, alpha, eij


def demo(seed: int) -> None:
    rng = np.random.default_rng(seed)
    batch, steps, features = 2, 4, 6
    # Row 0: last step is a clear outlier in the first feature.
    # Row 1: uniform-ish sequence so attention is closer to uniform.
    x = rng.normal(size=(batch, steps, features)).astype(np.float64) * 0.15
    x[0, 3] += np.array([2.5, 0, 0, 0, 0, 0])
    x[1] += 0.05
    w = np.zeros(features, dtype=np.float64)
    w[0] = 1.8  # attend to feature 0
    b = np.array([0.0, -0.15, 0.05, 0.1], dtype=np.float64)

    context, alpha, eij = attention(x, w, b)
    print("x shape            ", x.shape)
    print("W (attend feat 0)  ", np.round(w, 3))
    print("b (per timestep)   ", np.round(b, 3))
    print()
    print("eij = tanh(x·W + b)")
    print(np.round(eij, 4))
    print()
    print("alpha (softmax)")
    print(np.round(alpha, 4))
    print("alpha row sums     ", np.round(alpha.sum(axis=1), 6))
    print()
    print("context (batch, features)")
    print(np.round(context, 4))
    print()

    # Mask the outlier step on row 0. Mass should move to the first three.
    mask = np.ones((batch, steps), dtype=np.float64)
    mask[0, 3] = 0.0
    ctx_m, alpha_m, _ = attention(x, w, b, mask=mask)
    print("alpha after masking step 3 of row 0")
    print(np.round(alpha_m, 4))
    print("masked step mass   ", float(alpha_m[0, 3]))
    print("context[0] delta   ", np.round(ctx_m[0] - context[0], 4))
    print()

    # Empty-mask guard: all zeros should not NaN because of epsilon.
    dead = np.zeros((1, steps), dtype=np.float64)
    ctx_dead, alpha_dead, _ = attention(x[:1], w, b, mask=dead)
    print("all-masked alpha   ", np.round(alpha_dead, 6))
    print("all-masked finite  ", bool(np.isfinite(ctx_dead).all()))
    print()
    print(
        "This is the same scoring rule as attention_layer.py. The saved "
        "Keras models were built with steps=78 and features=512 "
        "(2 × 256 from the bidirectional LSTM)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    demo(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
