"""Numpy port of ``attention_layer.Attention``.

The Keras layer (Raffel et al. 2015) does:

    e_t = tanh(h_t · W + b_t)
    α   = softmax(e)          # epsilon in the denominator
    c   = Σ_t α_t h_t

``W`` is shared across time and has shape ``(features,)``.
``b`` is optional and has shape ``(steps,)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + np.finfo(exp.dtype).eps)


@dataclass
class AttentionWeights:
    W: np.ndarray
    b: np.ndarray | None = None

    def validate(self, steps: int, features: int) -> None:
        if self.W.shape != (features,):
            raise ValueError(f"W must have shape {(features,)}, got {self.W.shape}")
        if self.b is not None and self.b.shape != (steps,):
            raise ValueError(f"b must have shape {(steps,)}, got {self.b.shape}")


def attention_forward(
    hidden: np.ndarray,
    weights: AttentionWeights,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(context, alphas)``.

    ``hidden`` has shape ``(batch, steps, features)`` or ``(steps, features)``.
    ``mask`` if given is 1 for real tokens and 0 for pad, shape ``(batch, steps)``
    or ``(steps,)``.
    """
    squeezed = False
    if hidden.ndim == 2:
        hidden = hidden[np.newaxis, ...]
        squeezed = True
        if mask is not None and mask.ndim == 1:
            mask = mask[np.newaxis, ...]
    if hidden.ndim != 3:
        raise ValueError("hidden must be 2-d or 3-d")

    batch, steps, features = hidden.shape
    weights.validate(steps, features)

    # e_ij = (x · W) + b ; matches K.squeeze(K.dot(x, K.expand_dims(W)))
    scores = hidden @ weights.W
    if weights.b is not None:
        scores = scores + weights.b
    scores = np.tanh(scores)

    if mask is not None:
        if mask.shape != (batch, steps):
            raise ValueError("mask shape must match (batch, steps)")
        # Mask after the nonlinearity, before softmax, as the Keras layer does.
        scores = np.where(mask > 0, scores, -1e9)

    alphas = softmax(scores, axis=1)
    context = np.sum(hidden * alphas[..., np.newaxis], axis=1)

    if squeezed:
        return context[0], alphas[0]
    return context, alphas


def peaked_weights(steps: int, features: int, peak: int) -> AttentionWeights:
    """Build weights that put most mass on ``hidden[peak]`` when that step is large.

    ``W`` is all ones so the score of a step is ``tanh(sum(h_t) + b_t)``.
    ``b`` is zero except a positive bump at ``peak``.
    """
    if not 0 <= peak < steps:
        raise ValueError("peak is out of range")
    W = np.ones((features,), dtype=np.float64)
    b = np.zeros((steps,), dtype=np.float64)
    b[peak] = 2.5
    return AttentionWeights(W=W, b=b)
