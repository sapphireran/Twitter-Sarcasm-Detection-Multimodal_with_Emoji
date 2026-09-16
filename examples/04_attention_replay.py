#!/usr/bin/env python3
"""Replay the Raffel attention identities from attention_layer.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccs2lab.attention import raffel_attention, uniform_weights


def _close(name: str, got: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
    ok = np.allclose(got, expected, atol=atol)
    status = "ok" if ok else "FAIL"
    max_err = float(np.max(np.abs(got - expected)))
    print(f"[{status}] {name:40s}  max|err|={max_err:.2e}")
    if not ok:
        raise SystemExit(f"identity failed: {name}")


def main() -> int:
    rng = np.random.default_rng(2023)
    batch, steps, feat = 4, 8, 16

    # Identical hidden states + zero bias → uniform attention.
    hidden = np.broadcast_to(rng.normal(size=(1, 1, feat)), (batch, steps, feat)).copy()
    weight = rng.normal(size=(feat,))
    out = raffel_attention(hidden, weight)
    _close("uniform on identical rows", out.weights, uniform_weights(batch, steps))
    _close("context equals the repeated row", out.context, hidden[:, 0, :])

    # One step aligned with W, others orthogonal-ish → that step wins.
    hidden = np.zeros((1, steps, feat))
    hidden[0, 3, :] = weight / np.linalg.norm(weight)
    out = raffel_attention(hidden, weight)
    peak = int(out.weights.argmax(axis=1)[0])
    print(f"[ok] peak step (expected 3): {peak}  mass={out.weights[0, 3]:.3f}")
    if peak != 3:
        raise SystemExit("peak step was not 3")

    # Masking zeros a step after exp; remaining weights renormalize.
    hidden = rng.normal(size=(2, steps, feat))
    mask = np.ones((2, steps))
    mask[:, 0] = 0
    out = raffel_attention(hidden, weight, mask=mask)
    _close("masked step has zero weight", out.weights[:, 0], np.zeros((2,)))
    row_sum = out.weights.sum(axis=1)
    _close("masked rows still sum to 1", row_sum, np.ones((2,)))

    # tanh saturates: a huge timestep bias and a well-aligned hidden
    # state both score ≈ 1, so positional bias cannot drown content.
    hidden = np.zeros((1, steps, feat))
    hidden[0, 5, :] = weight / (np.linalg.norm(weight) + 1e-9)
    bias = np.zeros((steps,))
    bias[0] = 30.0
    saturated = raffel_attention(hidden, weight, bias=bias)
    print(
        f"[ok] tanh saturation: huge bias step0={saturated.weights[0, 0]:.3f} "
        f"vs aligned step5={saturated.weights[0, 5]:.3f}"
    )

    # With weak content, the (timesteps,) bias does win — this is the
    # positional prior the 2023 Keras layer can learn over pad length 78.
    weak = np.zeros((1, steps, feat))
    weak[0, 5, :] = 0.05 * weight / (np.linalg.norm(weight) + 1e-9)
    modest = np.zeros((steps,))
    modest[0] = 1.5
    biased = raffel_attention(weak, weight, bias=modest)
    print(
        f"[ok] weak content + modest bias: step0={biased.weights[0, 0]:.3f} "
        f"step5={biased.weights[0, 5]:.3f}"
    )
    if biased.weights[0, 0] <= biased.weights[0, 5]:
        raise SystemExit("modest bias should win against weak content")

    # Epsilon guard: all-masked row stays finite.
    mask = np.zeros((1, steps))
    out = raffel_attention(hidden[:1], weight, mask=mask)
    if not np.isfinite(out.context).all():
        raise SystemExit("all-masked context was not finite")
    print("[ok] all-masked row stays finite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
