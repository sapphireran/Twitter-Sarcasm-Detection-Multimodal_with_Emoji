"""NumPy port of ``attention_layer.Attention``.

The Keras layer scores each timestep with a shared vector ``W``, optional
per-timestep bias, ``tanh``, masked softmax, and a weighted sum. See
Raffel and Ellis, “Feed-Forward Networks with Attention Can Solve Some
Long-Term Memory Problems”, 2015 (https://arxiv.org/abs/1512.08756).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionOutput:
    context: np.ndarray
    weights: np.ndarray
    scores: np.ndarray


def masked_temporal_attention(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    epsilon: float = 1e-7,
) -> AttentionOutput:
    """Apply Raffel-style attention.

    Parameters
    ----------
    x:
        ``(batch, steps, features)``
    weights:
        ``(features,)`` shared score vector (``W`` in the Keras layer)
    bias:
        optional ``(steps,)`` vector added after the dot product
    mask:
        optional ``(batch, steps)`` 1/0 mask, applied *after* ``exp``
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x must be (batch, steps, features), got {x.shape}")
    if weights.shape != (x.shape[-1],):
        raise ValueError(
            f"weights must be ({x.shape[-1]},), got {weights.shape}"
        )

    scores = np.tensordot(x, weights, axes=([-1], [0]))
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (x.shape[1],):
            raise ValueError(
                f"bias must be ({x.shape[1]},), got {bias.shape}"
            )
        scores = scores + bias
    scores = np.tanh(scores)
    unnormalized = np.exp(scores)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != scores.shape:
            raise ValueError(
                f"mask must be {scores.shape}, got {mask.shape}"
            )
        unnormalized = unnormalized * mask
    denom = unnormalized.sum(axis=1, keepdims=True) + epsilon
    attention = unnormalized / denom
    context = (x * attention[..., None]).sum(axis=1)
    return AttentionOutput(context=context, weights=attention, scores=scores)


def random_attention_params(
    features: int,
    steps: int,
    *,
    use_bias: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    rng = rng or np.random.default_rng(0)
    w = rng.normal(scale=0.1, size=(features,))
    b = rng.normal(scale=0.01, size=(steps,)) if use_bias else None
    return w, b
