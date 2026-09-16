"""NumPy walkthrough of the Raffel attention layer used by the Bi-LSTM.

The Keras implementation lives in ``attention_layer.py``. It follows
Raffel et al., "Feed-Forward Networks with Attention Can Solve Some
Long-Term Memory Problems" (https://arxiv.org/abs/1512.08756):

1. score each time step as ``tanh(x_t · W + b_t)``
2. turn those scores into a mask-aware softmax
3. return the weighted sum over time, shape ``(batch, features)``

This module mirrors those three steps so the docs can print intermediate
arrays without loading TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = np.finfo(np.float32).eps


def softmax(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Row-wise softmax with the same epsilon guard as the Keras layer.

    The original layer does ``exp(score)`` then multiplies by the mask. We
    apply the mask before the max-shift so a padded step cannot dominate the
    normalization, then still add ``eps`` to the denominator.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
    # exp(-inf) is 0, so masked steps drop out of the partition function.
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / (np.sum(weights, axis=-1, keepdims=True) + EPS)


def raffel_attention(
    sequence: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the course project's attention pooling.

    Parameters
    ----------
    sequence:
        Array of shape ``(batch, steps, features)``.
    weights:
        Vector of shape ``(features,)`` — the Keras ``W`` parameter.
    bias:
        Optional vector of shape ``(steps,)``. The original layer learns one
        bias per time index, not per feature.
    mask:
        Optional 0/1 array of shape ``(batch, steps)``.
    """
    sequence = np.asarray(sequence, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError(f"sequence must be rank 3, got shape {sequence.shape}")
    if weights.shape != (sequence.shape[-1],):
        raise ValueError(
            f"weights must have shape {(sequence.shape[-1],)}, got {weights.shape}"
        )

    # e_ij = tanh(x · W [+ b])
    scores = np.tensordot(sequence, weights, axes=([-1], [0]))
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (sequence.shape[1],):
            raise ValueError(
                f"bias must have shape {(sequence.shape[1],)}, got {bias.shape}"
            )
        scores = scores + bias
    scores = np.tanh(scores)
    alphas = softmax(scores, mask=mask)
    weighted = sequence * alphas[..., None]
    return np.sum(weighted, axis=1), alphas, scores


@dataclass(frozen=True)
class AttentionWalkthrough:
    """Named intermediate tensors for the documentation example."""

    context: np.ndarray
    alphas: np.ndarray
    scores: np.ndarray

    def top_steps(self, batch_index: int = 0, k: int = 3) -> list[tuple[int, float]]:
        order = np.argsort(self.alphas[batch_index])[::-1]
        return [(int(i), float(self.alphas[batch_index, i])) for i in order[:k]]


def explain_attention(
    sequence: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> AttentionWalkthrough:
    context, alphas, scores = raffel_attention(sequence, weights, bias, mask)
    return AttentionWalkthrough(context=context, alphas=alphas, scores=scores)
