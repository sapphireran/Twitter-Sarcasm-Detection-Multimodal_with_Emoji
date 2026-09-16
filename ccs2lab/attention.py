"""NumPy port of the Raffel-style attention layer in ``attention_layer.py``.

The 2023 Keras layer follows Raffel et al. 2015
(https://arxiv.org/abs/1512.08756):

    e_t = tanh(h_t · W + b_t)
    a   = softmax(e)
    c   = sum_t a_t h_t

Two archive-specific details matter when you compare against Keras:

1. ``W`` is a vector of length ``features``, not a matrix. The score is
   a dot product, not a learned projection into a hidden attention size.
2. The optional bias is shaped ``(timesteps,)``, not ``(features,)``.
   That is unusual. It ties the layer to a fixed pad length (78 in the
   saved 2023 models) and gives each time index its own additive offset.

Masking happens *after* ``exp``, then the weights are renormalized with
``epsilon`` in the denominator to avoid NaNs when a row is all masked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionOutput:
    context: np.ndarray  # (batch, features)
    weights: np.ndarray  # (batch, steps)
    scores: np.ndarray  # (batch, steps) pre-softmax tanh scores


def raffel_attention(
    hidden: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    eps: float = 1e-7,
) -> AttentionOutput:
    """``hidden`` is ``(batch, steps, features)``; ``weight`` is ``(features,)``."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be 3D, got shape {hidden.shape}")
    if weight.ndim != 1 or weight.shape[0] != hidden.shape[-1]:
        raise ValueError(
            f"weight must be ({hidden.shape[-1]},), got {weight.shape}"
        )
    scores = np.tensordot(hidden, weight, axes=([-1], [0]))
    if bias is not None:
        if bias.shape != (hidden.shape[1],):
            raise ValueError(
                f"bias must be ({hidden.shape[1]},), got {bias.shape}"
            )
        scores = scores + bias
    scores = np.tanh(scores)
    unnorm = np.exp(scores)
    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError(
                f"mask must match scores {scores.shape}, got {mask.shape}"
            )
        unnorm = unnorm * mask.astype(unnorm.dtype, copy=False)
    weights = unnorm / (unnorm.sum(axis=1, keepdims=True) + eps)
    context = (hidden * weights[..., None]).sum(axis=1)
    return AttentionOutput(context=context, weights=weights, scores=scores)


def uniform_weights(batch: int, steps: int) -> np.ndarray:
    return np.full((batch, steps), 1.0 / steps, dtype=np.float64)
