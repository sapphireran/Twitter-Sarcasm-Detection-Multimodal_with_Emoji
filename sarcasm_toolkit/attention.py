"""Raffel-style temporal attention, matching ``attention_layer.py``.

The Keras layer from Raffel et al. (https://arxiv.org/abs/1512.08756) does:

1. ``e_t = tanh(x_t · W + b_t)``
2. ``a = softmax(e)`` (mask applied after the exp, then renormalized)
3. ``out = sum_t a_t x_t``

The original layer adds ``epsilon`` to the denominator so an all-masked
timestep does not produce NaNs. This module does the same, in pure
Python, so the attention example runs without TensorFlow.

Shapes:

* ``x``: ``(batch, steps, features)``
* ``W``: ``(features,)``
* ``b``: ``(steps,)`` or ``None``
* output: ``(batch, features)``
"""

from __future__ import annotations

import math
from typing import Sequence

EPSILON = 1e-7


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _tanh(value: float) -> float:
    # Clamp to keep math.exp stable for large |x|.
    clipped = max(-20.0, min(20.0, value))
    exp_neg = math.exp(-2.0 * clipped)
    return (1.0 - exp_neg) / (1.0 + exp_neg)


def softmax(scores: Sequence[float], mask: Sequence[float] | None = None) -> list[float]:
    """Masked softmax with the same epsilon guard as the Keras layer."""
    exps = [math.exp(score) for score in scores]
    if mask is not None:
        if len(mask) != len(exps):
            raise ValueError("mask length must match scores")
        exps = [value * float(flag) for value, flag in zip(exps, mask)]
    total = sum(exps) + EPSILON
    return [value / total for value in exps]


def attention_weights(
    sequence: Sequence[Sequence[float]],
    weights: Sequence[float],
    bias: Sequence[float] | None = None,
    mask: Sequence[float] | None = None,
) -> list[float]:
    """Return the attention distribution over one sequence."""
    if not sequence:
        return []
    n_features = len(sequence[0])
    if len(weights) != n_features:
        raise ValueError(
            f"W has {len(weights)} dims but sequence features are {n_features}"
        )
    energies: list[float] = []
    for step, frame in enumerate(sequence):
        if len(frame) != n_features:
            raise ValueError("ragged sequence; all timesteps need the same width")
        energy = _tanh(_dot(frame, weights) + (bias[step] if bias is not None else 0.0))
        energies.append(energy)
    return softmax(energies, mask=mask)


def attention_pool(
    batch: Sequence[Sequence[Sequence[float]]],
    weights: Sequence[float],
    bias: Sequence[float] | None = None,
    mask: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    """Pool a batch of sequences the same way ``Attention.call`` does.

    ``bias`` is length ``steps`` and is shared across the batch, matching
    the original Keras weight of shape ``(timesteps,)``.
    """
    pooled: list[list[float]] = []
    for index, sequence in enumerate(batch):
        step_mask = mask[index] if mask is not None else None
        alphas = attention_weights(sequence, weights, bias=bias, mask=step_mask)
        if not sequence:
            pooled.append([])
            continue
        width = len(sequence[0])
        out = [0.0] * width
        for alpha, frame in zip(alphas, sequence):
            for dim, value in enumerate(frame):
                out[dim] += alpha * value
        pooled.append(out)
    return pooled


def attention_report(
    sequence: Sequence[Sequence[float]],
    weights: Sequence[float],
    tokens: Sequence[str] | None = None,
    bias: Sequence[float] | None = None,
    mask: Sequence[float] | None = None,
) -> list[dict]:
    """Pair tokens with attention mass for the walkthrough example."""
    alphas = attention_weights(sequence, weights, bias=bias, mask=mask)
    rows: list[dict] = []
    for step, alpha in enumerate(alphas):
        token = tokens[step] if tokens is not None and step < len(tokens) else f"t{step}"
        rows.append({"index": step, "token": token, "weight": alpha})
    return rows
