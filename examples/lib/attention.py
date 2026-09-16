"""NumPy-free reimplementation of ``attention_layer.Attention``.

The Keras layer scores each time step with ``tanh(x_t · W + b_t)``, exponentiates,
masks, renormalizes, and returns the weighted sum. This module does the same
with only the standard library so the walkthrough (and the unit tests) do not
depend on TensorFlow.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dot product size mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def _add_scaled(vectors: Iterable[Sequence[float]], weights: Sequence[float]) -> List[float]:
    vecs = list(vectors)
    if not vecs:
        return []
    width = len(vecs[0])
    out = [0.0] * width
    for vec, weight in zip(vecs, weights):
        if len(vec) != width:
            raise ValueError("ragged sequence passed to temporal_attention")
        for i, value in enumerate(vec):
            out[i] += weight * value
    return out


def temporal_attention(
    sequence: Matrix,
    weight: Vector,
    bias: Optional[Vector] = None,
    mask: Optional[Sequence[float]] = None,
    epsilon: float = 1e-7,
) -> Tuple[List[float], List[float], List[float]]:
    """Return ``(pooled, alphas, pre_softmax_scores)``.

    ``sequence`` is ``(steps, features)`` — one example, no batch axis.
    ``weight`` is ``(features,)``. ``bias``, if given, is ``(steps,)``.
    ``mask`` is 1 for real tokens and 0 for pads.
    """
    steps = list(sequence)
    if not steps:
        raise ValueError("empty sequence")
    if bias is not None and len(bias) != len(steps):
        raise ValueError("bias length must equal the number of time steps")
    if mask is not None and len(mask) != len(steps):
        raise ValueError("mask length must equal the number of time steps")

    scores: List[float] = []
    for t, hidden in enumerate(steps):
        energy = _dot(hidden, weight)
        if bias is not None:
            energy += bias[t]
        scores.append(math.tanh(energy))

    exps = [math.exp(score) for score in scores]
    if mask is not None:
        exps = [value * float(mask[t]) for t, value in enumerate(exps)]
    denom = sum(exps) + epsilon
    alphas = [value / denom for value in exps]
    pooled = _add_scaled(steps, alphas)
    return pooled, alphas, scores


def argmax_token(tokens: Sequence[str], alphas: Sequence[float]) -> str:
    if not tokens or not alphas:
        raise ValueError("tokens and alphas must be non-empty")
    n = min(len(tokens), len(alphas))
    best = max(range(n), key=lambda i: alphas[i])
    return tokens[best]
