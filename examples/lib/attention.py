"""NumPy replay of the Keras attention layer in ``attention_layer.py``.

The course model stacked two bidirectional LSTMs with ``return_sequences=True``
and then applied Raffel-style attention (https://arxiv.org/abs/1512.08756):

1. score each time step with ``tanh(x W + b)``
2. softmax the scores (with an optional mask)
3. return the weighted sum of the hidden states

The original layer is a Keras 2 class. This module is the same math so
the architecture write-up can show attention weights on a toy sequence
without loading TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionOutput:
    context: np.ndarray
    weights: np.ndarray
    scores: np.ndarray


def _as_batch(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"expected (steps, features) or (batch, steps, features); got {array.shape}")
    return array


def attention_weights(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """Return softmax attention weights with shape ``(batch, steps)``."""

    return attention_forward(x, W, b=b, mask=mask, eps=eps).weights


def attention_forward(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    eps: float = 1e-7,
) -> AttentionOutput:
    """Match ``Attention.call`` from ``attention_layer.py``.

    Parameters
    ----------
    x:
        Hidden states, ``(steps, features)`` or ``(batch, steps, features)``.
    W:
        Feature-wise score vector of shape ``(features,)``.
    b:
        Optional per-step bias of shape ``(steps,)``. The Keras layer
        stored this as a weight of length ``input_shape[1]``.
    mask:
        Optional 0/1 mask broadcastable to ``(batch, steps)``.
    """

    batch = _as_batch(x)
    weights = np.asarray(W, dtype=np.float64).reshape(-1)
    if batch.shape[-1] != weights.shape[0]:
        raise ValueError(
            f"W has {weights.shape[0]} features but x has {batch.shape[-1]}"
        )

    scores = np.tensordot(batch, weights, axes=([-1], [0]))
    if b is not None:
        bias = np.asarray(b, dtype=np.float64).reshape(-1)
        if bias.shape[0] != batch.shape[1]:
            raise ValueError(
                f"b has {bias.shape[0]} steps but x has {batch.shape[1]}"
            )
        scores = scores + bias
    scores = np.tanh(scores)
    unnormalized = np.exp(scores)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=np.float64)
        if mask_array.ndim == 1:
            mask_array = np.broadcast_to(mask_array, unnormalized.shape)
        unnormalized = unnormalized * mask_array
    normalizer = unnormalized.sum(axis=1, keepdims=True) + eps
    alpha = unnormalized / normalizer
    context = (batch * alpha[..., np.newaxis]).sum(axis=1)
    return AttentionOutput(context=context, weights=alpha, scores=scores)


def demo_sequence(n_steps: int = 6, n_features: int = 8, seed: int = 7) -> dict[str, np.ndarray]:
    """Build a tiny sequence whose last two steps are scaled up.

    Useful for showing that attention can put mass on later tokens when
    ``W`` points along the scaled feature.
    """

    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.15, size=(n_steps, n_features))
    x[-2:] += 1.4
    W = np.ones(n_features) / np.sqrt(n_features)
    return {"x": x, "W": W, "b": np.zeros(n_steps)}
