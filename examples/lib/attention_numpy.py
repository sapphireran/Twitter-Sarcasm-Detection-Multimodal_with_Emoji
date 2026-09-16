"""NumPy clone of attention_layer.Attention.call.

Equations (Raffel et al. 2015, as implemented in the Keras layer):

    e = tanh(x @ W + b)          # b is optional and per-step
    α = exp(e) / (sum(exp(e)) + ε)
    h = sum_t α_t * x_t
"""

from __future__ import annotations

import numpy as np


def attention_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    eps: float = np.finfo(np.float64).eps,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (context, alpha).

    x:      (batch, steps, features)
    weights:(features,)
    bias:   (steps,) or None
    mask:   (batch, steps) bool / 0-1, True = keep
    """
    if x.ndim != 3:
        raise ValueError(f"expected 3-d x, got {x.shape}")
    if weights.shape != (x.shape[-1],):
        raise ValueError(f"W must be ({x.shape[-1]},), got {weights.shape}")

    scores = np.tensordot(x, weights, axes=([-1], [0]))  # (batch, steps)
    if bias is not None:
        if bias.shape != (x.shape[1],):
            raise ValueError(f"b must be ({x.shape[1]},), got {bias.shape}")
        scores = scores + bias
    scores = np.tanh(scores)
    unnormalized = np.exp(scores)
    if mask is not None:
        unnormalized = unnormalized * mask.astype(unnormalized.dtype)
    denom = unnormalized.sum(axis=1, keepdims=True) + eps
    alpha = unnormalized / denom
    context = (x * alpha[..., None]).sum(axis=1)
    return context, alpha
