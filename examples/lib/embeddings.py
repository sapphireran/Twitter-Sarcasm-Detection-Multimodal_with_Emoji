"""Tiny in-memory embedding tables that mimic the coursework fusion.

`data_utils.AverageVectorPerTweet` / `AverageVectorPerEmoji` average
every in-vocabulary row and fall back to a zero vector. `ml_read_data`
then concatenates the two means. This module does the same thing for
arbitrary tables so the docs example can run without GloVe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KeyedTable:
    """A dict-backed stand-in for gensim `KeyedVectors`."""

    vectors: dict[str, np.ndarray]
    dim: int

    def __post_init__(self) -> None:
        for key, row in self.vectors.items():
            arr = np.asarray(row, dtype=np.float64)
            if arr.shape != (self.dim,):
                raise ValueError(
                    f"{key!r} has shape {arr.shape}, expected ({self.dim},)"
                )
            self.vectors[key] = arr

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.vectors

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[key]

    @property
    def vocab(self) -> dict[str, np.ndarray]:
        # Coursework code uses `token in model.vocab`.
        return self.vectors


def average_rows(tokens: list[str], table: KeyedTable) -> np.ndarray:
    """Mean of in-vocabulary rows; zeros if nothing hits.

    Matches `AverageVectorPerTweet` / `AverageVectorPerEmoji`.
    """
    rows = [table[tok] for tok in tokens if tok in table]
    if not rows:
        return np.zeros((table.dim,), dtype=np.float64)
    return np.mean(np.stack(rows, axis=0), axis=0)


def fuse_modalities(
    tokens: list[str],
    word_table: KeyedTable,
    emoji_table: KeyedTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (word_mean, emoji_mean, concat)."""
    word_mean = average_rows(tokens, word_table)
    emoji_mean = average_rows(tokens, emoji_table)
    fused = np.concatenate([word_mean, emoji_mean], axis=0)
    return word_mean, emoji_mean, fused


def toy_glove() -> KeyedTable:
    """Hand-built 8-d table for the fusion walkthrough.

    Positive-polarity tokens point mostly along axis 0; complaint /
    negation tokens along axis 1; sarcasm hashtags along axis 2.
    """
    dim = 8
    rows = {
        "i": _one(dim, 3, 0.4),
        "love": _one(dim, 0, 1.0),
        "this": _one(dim, 3, 0.3),
        "walking": _one(dim, 3, 0.2),
        "to": _one(dim, 3, 0.1),
        "school": _one(dim, 3, 0.2),
        "getting": _one(dim, 1, 0.3),
        "home": _one(dim, 3, 0.2),
        "dirty": _one(dim, 1, 1.0),
        "bored": _one(dim, 1, 0.8),
        "want": _one(dim, 0, 0.2),
        "have": _one(dim, 3, 0.1),
        "someone": _one(dim, 3, 0.1),
        "speak": _one(dim, 3, 0.2),
        "#not": _one(dim, 2, 1.0),
        "#sarcastictweet": _one(dim, 2, 1.0),
        "#sarcasm": _one(dim, 2, 1.0),
        "loovee": _one(dim, 0, 0.9),
        "when": _one(dim, 3, 0.1),
        "people": _one(dim, 3, 0.2),
        "text": _one(dim, 3, 0.2),
        "back": _one(dim, 3, 0.1),
        "just": _one(dim, 3, 0.1),
        "imagined": _one(dim, 3, 0.2),
        "you": _one(dim, 3, 0.2),
        "dancing": _one(dim, 0, 0.4),
        "like": _one(dim, 0, 0.3),
    }
    return KeyedTable(rows, dim=dim)


def toy_emoji() -> KeyedTable:
    """Hand-built 8-d emoji table. Negative-face emoji sit on axis 0."""
    dim = 8
    rows = {
        "😒": _one(dim, 0, 1.0),
        "😑": _one(dim, 0, 0.8),
        "😩": _one(dim, 0, 0.9),
        "😅": _one(dim, 1, 0.4),
        "😭": _one(dim, 0, 0.7),
        "😄": _one(dim, 2, 1.0),
        "😃": _one(dim, 2, 0.8),
        "😌": _one(dim, 2, 0.6),
        "👏": _one(dim, 2, 0.5),
        "👎": _one(dim, 0, 0.9),
        "🔫": _one(dim, 0, 0.6),
        "💔": _one(dim, 0, 0.8),
    }
    return KeyedTable(rows, dim=dim)


def _one(dim: int, axis: int, value: float) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float64)
    vec[axis] = value
    return vec
