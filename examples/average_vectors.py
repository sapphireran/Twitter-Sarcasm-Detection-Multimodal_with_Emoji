"""Mean-pool a tweet the way ``data_utils.AverageVectorPer*`` does.

Empty in-vocabulary sets become a zero vector of the table's dimension.
That is the 2023 behavior: a tweet with no emoji still has a valid 400-d
row whose second half is all zeros.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .hash_embeddings import HashEmbeddings


def average_channel(
    tokens: Sequence[str], table: HashEmbeddings
) -> np.ndarray:
    rows = [table.embed(token) for token in tokens if table.known(token)]
    if not rows:
        return np.zeros((table.dim,), dtype=np.float64)
    stacked = np.stack(rows, axis=0)
    return stacked.mean(axis=0)


def multimodal_features(
    tokenized: Sequence[Sequence[str]],
    text_table: HashEmbeddings,
    emoji_table: HashEmbeddings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_text, X_multi)`` with shapes ``(n, d)`` and ``(n, 2d)``."""
    text_rows = [average_channel(tokens, text_table) for tokens in tokenized]
    emoji_rows = [average_channel(tokens, emoji_table) for tokens in tokenized]
    x_text = np.stack(text_rows, axis=0)
    x_emoji = np.stack(emoji_rows, axis=0)
    x_multi = np.concatenate([x_text, x_emoji], axis=1)
    return x_text, x_multi
