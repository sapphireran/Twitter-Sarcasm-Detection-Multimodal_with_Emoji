"""NumPy port of the course project's temporal attention layer.

``attention_layer.Attention`` follows Raffel et al., 2016
(https://arxiv.org/abs/1512.08756). After a bidirectional LSTM emits a
sequence ``x`` of shape ``(batch, steps, features)`` the layer learns a
score for each step, turns those scores into a softmax, and returns the
weighted sum.

This module is intentionally independent of TensorFlow so the attention
walkthrough can run in a tiny environment.
"""

from __future__ import annotations

import numpy as np


def tanh(values: np.ndarray) -> np.ndarray:
    return np.tanh(values)


def masked_softmax(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Row-wise softmax with the same epsilon guard as the Keras layer."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(shifted)
    if mask is not None:
        weights = weights * mask.astype(weights.dtype)
    totals = np.sum(weights, axis=1, keepdims=True) + np.finfo(weights.dtype).eps
    return weights / totals


def attention_scores(
    sequences: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``tanh(x W + b)`` for a batch of sequences.

    Parameters
    ----------
    sequences:
        Array of shape ``(batch, steps, features)``.
    weight:
        Vector of shape ``(features,)``.
    bias:
        Optional vector of shape ``(steps,)``, matching the original layer
        which stored a per-timestep bias rather than a per-feature bias.
    """
    if sequences.ndim != 3:
        raise ValueError(f"expected a 3D batch, got shape {sequences.shape}")
    if weight.ndim != 1 or weight.shape[0] != sequences.shape[-1]:
        raise ValueError(
            f"weight shape {weight.shape} does not match features {sequences.shape[-1]}"
        )

    raw = np.matmul(sequences, weight)
    if bias is not None:
        if bias.shape != (sequences.shape[1],):
            raise ValueError(
                f"bias shape {bias.shape} does not match steps {sequences.shape[1]}"
            )
        raw = raw + bias
    return tanh(raw)


def attention_pool(
    sequences: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(context, attention_weights)``.

    ``context`` has shape ``(batch, features)``. ``attention_weights`` has
    shape ``(batch, steps)``.
    """
    scores = attention_scores(sequences, weight, bias=bias)
    weights = masked_softmax(scores, mask=mask)
    context = np.sum(sequences * weights[..., None], axis=1)
    return context, weights


def uniform_attention_pool(sequences: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean-pool a sequence. Useful as a no-attention baseline."""
    if mask is None:
        return np.mean(sequences, axis=1)
    widths = mask.sum(axis=1, keepdims=True)
    widths = np.maximum(widths, 1.0)
    return (sequences * mask[..., None]).sum(axis=1) / widths
