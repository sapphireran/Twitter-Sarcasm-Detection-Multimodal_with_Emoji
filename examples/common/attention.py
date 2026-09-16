"""NumPy reimplementation of ``attention_layer.Attention``.

The Keras layer (Raffel et al., 2016) maps a sequence of hidden states
``(batch, steps, features)`` to a context vector ``(batch, features)``:

1. score each step as ``e_t = tanh(h_t · W + b)``
2. mask (optional)
3. ``a = softmax(e)`` with an epsilon in the denominator to avoid 0/0
4. return the weighted sum ``sum_t a_t h_t``

The examples use this copy so we can inspect attention weights without loading
TensorFlow. Numerical constants (Glorot-scale ``W``, learned ``b``) are passed
in by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AdditiveAttention:
    """Raffel-style attention over a time axis."""

    w: np.ndarray  # (features,)
    bias: Optional[np.ndarray] = None  # (steps,) or None
    epsilon: float = np.finfo(np.float32).eps

    def scores(self, hidden: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the pre-softmax energies ``e`` of shape ``(batch, steps)``."""
        if hidden.ndim != 3:
            raise ValueError("hidden must be (batch, steps, features)")
        energy = np.tanh(np.dot(hidden, self.w))
        if self.bias is not None:
            energy = energy + self.bias
        if mask is not None:
            # Mask is applied *after* exp in the Keras layer; we keep the same
            # contract here by leaving zeros that will be multiplied post-exp.
            pass
        return energy

    def weights(self, hidden: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        energy = self.scores(hidden, mask=mask)
        alpha = np.exp(energy)
        if mask is not None:
            alpha = alpha * mask.astype(alpha.dtype)
        denom = np.sum(alpha, axis=1, keepdims=True) + self.epsilon
        return alpha / denom

    def context(self, hidden: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        alpha = self.weights(hidden, mask=mask)
        return np.sum(hidden * alpha[..., None], axis=1)


def random_attention(features: int, steps: int, rng: np.random.Generator) -> AdditiveAttention:
    """Glorot-uniform ``W`` as in the original layer, zero bias."""
    limit = np.sqrt(6.0 / features)
    w = rng.uniform(-limit, limit, size=(features,)).astype(np.float32)
    bias = np.zeros((steps,), dtype=np.float32)
    return AdditiveAttention(w=w, bias=bias)
