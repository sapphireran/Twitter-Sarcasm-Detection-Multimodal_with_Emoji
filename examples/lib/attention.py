"""NumPy replay of ``attention_layer.Attention.call``.

The Keras layer scores each timestep with ``tanh(x W + b)``, masks after
``exp``, and returns the weighted sum over time. See docs/attention.md.
"""

from __future__ import annotations

import numpy as np

# Keras backend epsilon is 1e-7 in the TF 2 builds used in 2023.
KERAS_EPSILON = 1e-7


def temporal_attention(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    epsilon: float = KERAS_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Raffel-style feed-forward attention.

    Parameters
    ----------
    x:
        Hidden sequence, shape ``(batch, steps, features)``.
    weight:
        Score vector ``W``, shape ``(features,)``.
    bias:
        Optional per-timestep bias, shape ``(steps,)``.
    mask:
        Optional 1/0 mask, shape ``(batch, steps)``. Zero means ignore.
    epsilon:
        Added to the softmax denominator, matching ``K.epsilon()``.

    Returns
    -------
    context:
        Weighted sum, shape ``(batch, features)``.
    weights:
        Softmax weights, shape ``(batch, steps)``.
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x must be (batch, steps, features), got {x.shape}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"W must have shape {(x.shape[-1],)}, got {weight.shape}")

    scores = np.tanh(x @ weight)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (x.shape[1],):
            raise ValueError(f"bias must have shape {(x.shape[1],)}, got {bias.shape}")
        scores = scores + bias

    unnormalised = np.exp(scores)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float64)
        if mask_arr.shape != scores.shape:
            raise ValueError(f"mask must have shape {scores.shape}, got {mask_arr.shape}")
        unnormalised = unnormalised * mask_arr

    weights = unnormalised / (unnormalised.sum(axis=1, keepdims=True) + epsilon)
    context = (x * weights[..., None]).sum(axis=1)
    return context, weights
