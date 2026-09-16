"""NumPy port of the Raffel-style attention in ``attention_layer.py``."""

from __future__ import annotations

import math

import numpy as np


def softmax(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Softmax over the last axis, with the same epsilon guard as Keras."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    if mask is not None:
        weights = weights * mask
    denom = weights.sum(axis=-1, keepdims=True) + np.finfo(weights.dtype).eps
    return weights / denom


def raffel_attention(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(context, attention_weights)``.

    Parameters
    ----------
    x:
        ``(batch, steps, features)`` — the bidirectional LSTM sequences.
    weight:
        ``(features,)`` — the ``W`` vector in ``attention_layer.py``.
    bias:
        Optional ``(steps,)`` per-timestep bias. The Keras layer stores a
        bias of this shape, so padded length is baked into the layer.
    mask:
        Optional ``(batch, steps)`` float mask (1 = keep, 0 = drop).
    """
    if x.ndim != 3:
        raise ValueError(f"expected (batch, steps, features), got {x.shape}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"weight shape {weight.shape} != ({x.shape[-1]},)")
    # e_ij = tanh(x · W [+ b])
    scores = np.tanh(np.dot(x, weight))
    if bias is not None:
        if bias.shape != (x.shape[1],):
            raise ValueError(f"bias shape {bias.shape} != ({x.shape[1]},)")
        scores = scores + bias
    weights = softmax(scores, mask=mask)
    context = (x * weights[..., None]).sum(axis=1)
    return context, weights


def make_demo_sequence(n_steps: int = 8, n_features: int = 6, seed: int = 7) -> dict[str, np.ndarray]:
    """Synthetic sequence with a late 'cue' spike, like ``#not`` at the end."""
    rng = np.random.default_rng(seed)
    x = 0.15 * rng.standard_normal((1, n_steps, n_features))
    # Early tokens are a bland positive stem; last two steps are the reversal.
    x[0, 0] += np.array([1.2, 0.8, 0.0, 0.0, 0.0, 0.0])
    x[0, 1] += np.array([1.0, 0.6, 0.0, 0.0, 0.0, 0.0])
    x[0, -2] += np.array([0.0, 0.0, 0.0, 1.4, 1.1, 0.0])
    x[0, -1] += np.array([0.0, 0.0, 0.0, 1.6, 1.3, 0.2])
    weight = np.array([0.1, 0.1, 0.0, 1.2, 1.1, 0.3], dtype=float)
    bias = np.zeros(n_steps)
    mask = np.ones((1, n_steps))
    mask[0, 4:6] = 0.0  # pretend two pad slots in the middle
    context, attn = raffel_attention(x, weight, bias=bias, mask=mask)
    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "mask": mask,
        "context": context,
        "attention": attn,
    }


def attention_entropy(weights: np.ndarray) -> np.ndarray:
    """Higher entropy = flatter attention. Useful as a sanity check."""
    clipped = np.clip(weights, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def expected_step(weights: np.ndarray) -> np.ndarray:
    steps = np.arange(weights.shape[-1])
    return (weights * steps).sum(axis=-1)


def _self_check() -> None:
    demo = make_demo_sequence()
    attn = demo["attention"][0]
    if not math.isclose(float(attn.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"attention weights must sum to 1, got {attn.sum()}")
    if attn[4] > 1e-8 or attn[5] > 1e-8:
        raise RuntimeError("masked steps should have ~0 weight")
    if attn[-1] + attn[-2] < attn[0] + attn[1]:
        raise RuntimeError("cue steps should dominate the bland opener")


if __name__ == "__main__":
    _self_check()
    print("attention.py self-check ok")
