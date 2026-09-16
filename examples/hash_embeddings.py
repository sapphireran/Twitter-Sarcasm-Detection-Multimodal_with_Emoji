"""Deterministic stand-ins for GloVe Twitter 200-d and emoji2vec.

The 2023 notebooks look tokens up in ``gensim.KeyedVectors``. Shipping those
tables is impossible in a docs PR, so each token is hashed into a stable
vector instead.

Two namespaces (``text`` and ``emoji``) keep the channels from accidentally
sharing geometry, which is the same separation ``ml_read_data`` gets from
loading two different binaries.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from .tokenize import is_emoji_token

DEFAULT_DIM = 32


def _stable_vec(namespace: str, token: str, dim: int) -> np.ndarray:
    material = f"{namespace}::{token}".encode("utf-8")
    # Enough digest bytes for a 200-d table if a caller asks for one.
    needed = dim * 4
    buf = bytearray()
    counter = 0
    while len(buf) < needed:
        buf.extend(hashlib.sha256(material + counter.to_bytes(2, "little")).digest())
        counter += 1
    ints = np.frombuffer(bytes(buf[:needed]), dtype=np.int32).astype(np.float64)
    vec = (ints % 2000) / 1000.0 - 1.0  # roughly [-1, 1)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class HashEmbeddings:
    """Tiny keyed-vectors lookalike with a ``.vocab`` set and ``[token]``."""

    def __init__(self, namespace: str, dim: int = DEFAULT_DIM) -> None:
        if dim < 4:
            raise ValueError("dim must be at least 4")
        self.namespace = namespace
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def __contains__(self, token: str) -> bool:
        return True

    @property
    def vocab(self) -> set[str]:
        # Gensim 3 used ``.vocab``. We expose the cache plus a sentinel so
        # ``token in model.vocab`` stays meaningful for tokens we have seen.
        return set(self._cache)

    def known(self, token: str) -> bool:
        """Text table skips emoji; emoji table skips non-emoji."""
        emoji = is_emoji_token(token)
        if self.namespace == "emoji":
            return emoji
        return not emoji

    def embed(self, token: str) -> np.ndarray:
        if not self.known(token):
            raise KeyError(f"{token!r} is outside the {self.namespace} table")
        cached = self._cache.get(token)
        if cached is None:
            cached = _stable_vec(self.namespace, token, self.dim)
            self._cache[token] = cached
        return cached

    def __getitem__(self, token: str) -> np.ndarray:
        return self.embed(token)


def default_tables(dim: int = DEFAULT_DIM) -> tuple[HashEmbeddings, HashEmbeddings]:
    return HashEmbeddings("text", dim=dim), HashEmbeddings("emoji", dim=dim)


def embed_tokens(
    tokens: Iterable[str], table: HashEmbeddings
) -> list[np.ndarray]:
    return [table.embed(token) for token in tokens if table.known(token)]
