"""Early-fusion helpers that mirror ``ml_read_data``'s concatenate step."""

from __future__ import annotations

import numpy as np


def concat_channels(text: np.ndarray, emoji: np.ndarray) -> np.ndarray:
    if text.shape[-1] != emoji.shape[-1]:
        raise ValueError("text and emoji channels must share a width")
    return np.concatenate([text, emoji], axis=-1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def channel_norms(multi: np.ndarray) -> tuple[float, float]:
    """L2 norms of the first and second halves of a concatenated row."""
    if multi.ndim != 1 or multi.size % 2 != 0:
        raise ValueError("expected a 1-d even-width multimodal vector")
    half = multi.size // 2
    text = float(np.linalg.norm(multi[:half]))
    emoji = float(np.linalg.norm(multi[half:]))
    return text, emoji


def pairwise_cosine_matrix(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = rows / norms
    return unit @ unit.T
