"""NumPy copy of the Raffel-style attention layer in ``attention_layer.py``.

The Keras layer sits on a sequence encoder (here, stacked BiLSTMs with
``return_sequences=True``) and collapses time:

1. Score each step: ``e_t = tanh(x_t · W + b_t)``.
2. Softmax the scores, optionally multiplying by a padding mask first.
3. Return the weighted sum of the step vectors.

Paper: Raffel & Ellis, *Feed-Forward Networks with Attention Can Solve
Some Long-Term Memory Problems*, 2015. https://arxiv.org/abs/1512.08756
"""

from __future__ import annotations

import numpy as np


def attention_scores(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``tanh(x W + b)`` scores of shape ``(batch, steps)``."""
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x must be (batch, steps, features), got {x.shape}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(
            f"weight must be ({x.shape[-1]},), got {weight.shape}"
        )
    scores = np.dot(x, weight)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (x.shape[1],):
            raise ValueError(
                f"bias must be ({x.shape[1]},) to match steps, got {bias.shape}"
            )
        scores = scores + bias
    return np.tanh(scores)


def masked_softmax(
    scores: np.ndarray,
    mask: np.ndarray | None = None,
    epsilon: float = np.finfo(np.float64).eps,
) -> np.ndarray:
    """Softmax over time with the same epsilon guard as the Keras layer.

    The original comment in ``attention_layer.py`` notes that an all-zero
    (or all-masked) row would otherwise divide by zero and produce NaNs.
    """
    weights = np.exp(np.asarray(scores, dtype=np.float64))
    if mask is not None:
        weights = weights * np.asarray(mask, dtype=np.float64)
    denom = np.sum(weights, axis=1, keepdims=True) + epsilon
    return weights / denom


def attention_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    epsilon: float = np.finfo(np.float64).eps,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(context, attention_weights)``.

    ``context`` has shape ``(batch, features)``.
    ``attention_weights`` has shape ``(batch, steps)`` and sums to 1 along
    time (up to the epsilon used in the denominator).
    """
    x = np.asarray(x, dtype=np.float64)
    scores = attention_scores(x, weight, bias=bias)
    attn = masked_softmax(scores, mask=mask, epsilon=epsilon)
    context = np.sum(x * attn[..., None], axis=1)
    return context, attn


def uniform_attention(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Baseline: mean-pool (optionally masked). Useful in demos."""
    x = np.asarray(x, dtype=np.float64)
    if mask is None:
        return x.mean(axis=1)
    mask = np.asarray(mask, dtype=np.float64)
    denom = mask.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return (x * mask[..., None]).sum(axis=1) / denom
