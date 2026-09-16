"""Mean-pool and concat, mirroring AverageVectorPerTweet / AverageVectorPerEmoji."""

from __future__ import annotations

import numpy as np


def lookup_or_skip(tokens: list[str], table: dict[str, np.ndarray], dim: int) -> np.ndarray:
    rows = [table[tok] for tok in tokens if tok in table]
    if not rows:
        return np.zeros((dim,), dtype=np.float64)
    stacked = np.asarray(rows, dtype=np.float64)
    return stacked.mean(axis=0)


def average_rows(token_lists: list[list[str]], table: dict[str, np.ndarray], dim: int) -> np.ndarray:
    return np.stack([lookup_or_skip(tokens, table, dim) for tokens in token_lists], axis=0)


def concat_modalities(word_avg: np.ndarray, emoji_avg: np.ndarray) -> np.ndarray:
    if word_avg.shape[0] != emoji_avg.shape[0]:
        raise ValueError("row counts must match")
    return np.concatenate([word_avg, emoji_avg], axis=1)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
