"""NumPy port of ``attention_layer.Attention``.

The Keras layer scores each timestep with ``tanh(x · W + b)``, masks after
the exponential, and returns the weighted sum of hidden states. This module
exposes the scores, the normalized weights, and the context vector so a
walkthrough can print them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AttentionResult:
    scores: np.ndarray
    weights: np.ndarray
    context: np.ndarray


def _as_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[np.newaxis, ...]
    if x.ndim != 3:
        raise ValueError(f"expected (T, H) or (B, T, H), got {x.shape}")
    return x


def raffel_attention(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    epsilon: float = 1e-7,
) -> AttentionResult:
    """Compute Raffel attention.

    Parameters
    ----------
    x:
        ``(T, H)`` or ``(B, T, H)``.
    weight:
        ``(H,)`` — the Keras ``W`` vector.
    bias:
        Optional ``(T,)`` per-timestep bias, matching the 2023 layer.
    mask:
        Optional boolean / 0-1 array broadcastable to ``(B, T)``. True /
        1 keeps the timestep.
    """

    batch = _as_batch(np.asarray(x, dtype=np.float64))
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    hidden = batch.shape[-1]
    if w.shape != (hidden,):
        raise ValueError(f"W must have shape ({hidden},), got {w.shape}")

    scores = np.tensordot(batch, w, axes=([-1], [0]))
    if bias is not None:
        b = np.asarray(bias, dtype=np.float64).reshape(-1)
        if b.shape[0] != batch.shape[1]:
            raise ValueError(
                f"bias length {b.shape[0]} does not match timesteps {batch.shape[1]}"
            )
        scores = scores + b
    scores = np.tanh(scores)

    exp = np.exp(scores)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float64)
        if mask_arr.ndim == 1:
            mask_arr = np.broadcast_to(mask_arr, scores.shape)
        elif mask_arr.ndim == 2 and mask_arr.shape != scores.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match {scores.shape}")
        exp = exp * mask_arr

    denom = exp.sum(axis=1, keepdims=True) + epsilon
    weights = exp / denom
    context = (batch * weights[..., np.newaxis]).sum(axis=1)
    return AttentionResult(scores=scores, weights=weights, context=context)


def uniform_pool(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean pool, optionally ignoring padded steps. Contrast class for attention."""

    batch = _as_batch(np.asarray(x, dtype=np.float64))
    if mask is None:
        return batch.mean(axis=1)
    mask_arr = np.asarray(mask, dtype=np.float64)
    if mask_arr.ndim == 1:
        mask_arr = np.broadcast_to(mask_arr, batch.shape[:2])
    weights = mask_arr / (mask_arr.sum(axis=1, keepdims=True) + 1e-7)
    return (batch * weights[..., np.newaxis]).sum(axis=1)


def demo_sequence() -> np.ndarray:
    """A 5-step, 4-d sequence: sincere setup, then a sarcasm-like spike."""

    return np.array(
        [
            [0.8, 0.1, 0.0, 0.0],  # "I"
            [0.7, 0.2, 0.0, 0.1],  # "love"
            [0.1, 0.8, 0.1, 0.0],  # "monday"
            [0.0, 0.1, 0.9, 0.2],  # "mornings"
            [0.0, 0.0, 0.1, 1.4],  # "#not" / 😒
        ],
        dtype=np.float64,
    )


def demo_weight() -> np.ndarray:
    """Looks at the last hidden unit — the synthetic sarcasm dimension."""

    return np.array([0.05, 0.05, 0.10, 1.20], dtype=np.float64)
