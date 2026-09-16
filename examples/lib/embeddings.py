"""Tiny stand-ins for the GloVe + emoji2vec averaging in ``data_utils.py``.

The course notebooks load:

* ``glove.twitter.27B.200d.bin`` — 200-d Twitter GloVe (not in this repo)
* ``emoji2vec_twitter.bin`` — emoji vectors aligned to that space

``AverageVectorPerTweet`` / ``AverageVectorPerEmoji`` then mean-pool every
in-vocabulary token and concatenate the two 200-d means for the
multi-modal classical models.

This module repeats that recipe with a *dummy* keyed-vector table so the
pipeline can be walked through on a laptop without the 1.2M-word dump.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tokenize import is_emoji, tokenize_tweet


@dataclass
class DummyKeyedVectors:
    """Minimal ``gensim.models.KeyedVectors`` look-alike."""

    table: dict[str, np.ndarray]
    dim: int

    @classmethod
    def random_from_tokens(
        cls,
        tokens: list[str],
        *,
        dim: int = 16,
        seed: int = 0,
        include_unknown: bool = False,
    ) -> "DummyKeyedVectors":
        rng = np.random.default_rng(seed)
        table: dict[str, np.ndarray] = {}
        for token in tokens:
            if token not in table:
                vector = rng.normal(scale=0.25, size=dim)
                vector = vector / (np.linalg.norm(vector) + 1e-9)
                table[token] = vector
        if include_unknown:
            table["<unk>"] = np.zeros(dim)
        return cls(table=table, dim=dim)

    def __contains__(self, token: str) -> bool:
        return token in self.table

    def __getitem__(self, token: str) -> np.ndarray:
        return self.table[token]

    @property
    def vocab(self) -> dict[str, np.ndarray]:
        """Gensim-2-style ``.vocab`` attribute used in ``data_utils.py``."""

        return self.table


def average_vectors(
    docs: list[list[str]],
    model: DummyKeyedVectors,
    *,
    predicate=None,
) -> np.ndarray:
    """Mean-pool in-vocabulary tokens, else a zero vector.

    ``predicate`` lets the emoji path keep only emoji tokens, matching
    ``AverageVectorPerEmoji``.
    """

    rows: list[np.ndarray] = []
    for tokens in docs:
        chosen = []
        for token in tokens:
            if predicate is not None and not predicate(token):
                continue
            if token in model:
                chosen.append(model[token])
        if chosen:
            rows.append(np.mean(np.stack(chosen, axis=0), axis=0))
        else:
            rows.append(np.zeros(model.dim, dtype=np.float64))
    return np.stack(rows, axis=0)


def concatenate_modalities(word_avg: np.ndarray, emoji_avg: np.ndarray) -> np.ndarray:
    """The 400-d multi-modal feature used by the sklearn baselines."""

    if word_avg.shape[0] != emoji_avg.shape[0]:
        raise ValueError("word and emoji matrices must have the same batch size")
    return np.concatenate([word_avg, emoji_avg], axis=1)


def build_padded_sequences(
    docs: list[list[str]],
    *,
    maxlen: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Keras-style integer encoding with post-padding.

    ``data_utils.Preprocess`` fitted a ``Tokenizer``, turned texts into
    sequences, then ``pad_sequences(..., padding='post')``. Index ``0`` is
    reserved for pad; ``1..N`` are terms in first-seen order.
    """

    word_index: dict[str, int] = {}
    encoded: list[list[int]] = []
    for tokens in docs:
        row: list[int] = []
        for token in tokens:
            if token not in word_index:
                word_index[token] = len(word_index) + 1
            row.append(word_index[token])
        encoded.append(row)
    if maxlen is None:
        maxlen = max((len(row) for row in encoded), default=0)
    padded = np.zeros((len(encoded), maxlen), dtype=np.int32)
    for i, row in enumerate(encoded):
        clipped = row[:maxlen]
        padded[i, : len(clipped)] = clipped
    return padded, word_index


def tokenize_docs(texts: list[str]) -> list[list[str]]:
    return [tokenize_tweet(text) for text in texts]


def emoji_token_predicate(token: str) -> bool:
    return any(is_emoji(char) for char in token)
