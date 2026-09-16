#!/usr/bin/env python3
"""NumPy Raffel attention: weights sum to 1, masks drop a step, output is Σ a x.

Matches the formula in ``attention_layer.Attention.call``:

    e = tanh(x · W + b)
    a = softmax(e)          # mask after exp; +eps in the denominator
    h = Σ a_t x_t
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import raffel_attention  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(2023)
    batch, steps, features = 2, 3, 4
    x = rng.normal(size=(batch, steps, features))
    # Make step 1 of item 0 a very large-magnitude vector so attention
    # should put most of its mass there when W aligns with it.
    x[0, 1] = np.array([3.0, 0.0, 0.0, 0.0])
    weight = np.array([1.0, 0.0, 0.0, 0.0])
    bias = np.zeros((steps,))

    context, weights = raffel_attention(x, weight, bias)
    print("unmasked weights")
    print(np.round(weights, 4))
    print("row sums", np.round(weights.sum(axis=1), 6))
    print("context[0] vs x[0,1] (should be close):")
    print("  context", np.round(context[0], 4))
    print("  x[0,1] ", np.round(x[0, 1], 4))

    mask = np.ones((batch, steps))
    mask[0, 1] = 0.0  # drop the spike
    ctx_m, w_m = raffel_attention(x, weight, bias, mask=mask)
    print()
    print("masked weights (item 0, step 1 forced off)")
    print(np.round(w_m, 4))
    if w_m[0, 1] > 1e-6:
        raise SystemExit("mask failed: mass leaked onto a masked step")
    if not np.allclose(w_m.sum(axis=1), 1.0, atol=1e-6):
        raise SystemExit("masked weights do not sum to 1")

    # Reconstruction check: h == Σ a x
    recon = (x * weights[..., None]).sum(axis=1)
    if not np.allclose(recon, context):
        raise SystemExit("context is not the weighted sum")
    print()
    print("checks passed: mask zeros a step, rows sum to 1, h = Σ a x")
    print(f"masked context[0] {np.round(ctx_m[0], 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
