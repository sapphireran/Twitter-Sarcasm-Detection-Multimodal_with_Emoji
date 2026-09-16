"""NumPy Raffel attention, matching `attention_layer.Attention.call`.

For `x` of shape (batch, steps, features):

    e = tanh(x · W + b)          # b is per-timestep if present
    α = softmax(e)               # mask applied after exp
    out = Σ_t α_t x_t            # (batch, features)
"""

from __future__ import annotations

import numpy as np


def raffel_attention(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    epsilon: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(context, alpha)`.

    Parameters
    ----------
    x:
        `(batch, steps, features)`.
    weight:
        `(features,)` — the layer's `W`.
    bias:
        Optional `(steps,)` vector added to the energy before `tanh`.
    mask:
        Optional `(batch, steps)` 0/1 mask (1 = keep). Applied after
        `exp`, same as the Keras layer.
    """
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"x must be 3-d, got {x.shape}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(
            f"weight must have shape ({x.shape[-1]},), got {weight.shape}"
        )

    energy = np.tensordot(x, weight, axes=([-1], [0]))  # (batch, steps)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (x.shape[1],):
            raise ValueError(
                f"bias must have shape ({x.shape[1]},), got {bias.shape}"
            )
        energy = energy + bias
    energy = np.tanh(energy)

    alpha = np.exp(energy)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != energy.shape:
            raise ValueError(
                f"mask must have shape {energy.shape}, got {mask.shape}"
            )
        alpha = alpha * mask
    denom = np.sum(alpha, axis=1, keepdims=True) + epsilon
    alpha = alpha / denom

    context = np.sum(x * alpha[..., None], axis=1)
    return context, alpha


def demo_sequence() -> dict[str, np.ndarray]:
    """A 1-tweet toy sequence used by the walkthrough script.

    Steps: i / love / this / 😒 / #not
    Features are 4-d so you can read them in the printed table.
    The last two steps carry the sarcasm signal on axis 0.
    """
    tokens = np.array(["i", "love", "this", "😒", "#not"])
    x = np.array(
        [
            [
                [0.10, 0.80, 0.00, 0.10],  # i
                [0.05, 0.95, 0.00, 0.00],  # love  (positive)
                [0.10, 0.40, 0.10, 0.10],  # this
                [0.90, 0.05, 0.40, 0.00],  # 😒     (emoji)
                [0.95, 0.00, 0.00, 0.20],  # #not   (tag)
            ]
        ],
        dtype=np.float64,
    )
    # W looks at axis 0 — the sarcasm / emoji channel.
    weight = np.array([1.4, -0.2, 0.1, 0.0], dtype=np.float64)
    bias = np.array([0.0, 0.0, 0.0, 0.05, 0.10], dtype=np.float64)
    mask = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    return {"tokens": tokens, "x": x, "weight": weight, "bias": bias, "mask": mask}
