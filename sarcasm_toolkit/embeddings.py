"""Tiny embedding helpers for the attention walkthrough.

The 2023 notebooks load ``glove.twitter.27B.200d`` (~1.2 GB) plus
``emoji2vec``. Those files are not required to explain *how* average
pooling and attention pooling differ. This module builds a deterministic
toy table from a handful of polarity / cue tokens so examples stay
offline and fast.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

TOY_DIM = 8

# Hand-set axes so the attention demo is readable:
#   0: positive valence
#   1: negative valence
#   2: sarcasm-cue / negation
#   3: emoji / affect
#   4: social (@user)
#   remaining dims: small hash noise so unknown tokens are not all-zero
SEED_VECTORS: dict[str, list[float]] = {
    "love": [1.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0],
    "great": [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
    "best": [0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
    "happy": [0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1],
    "yay": [0.7, 0.0, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0],
    "hate": [0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0],
    "worst": [0.0, 0.95, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    "annoyed": [0.0, 0.7, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0],
    "late": [0.0, 0.4, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1],
    "#not": [0.1, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "#sarcasm": [0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    "#sarcastictweet": [0.0, 0.0, 0.95, 0.1, 0.0, 0.0, 0.0, 0.0],
    "#yeahright": [0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.0],
    "not": [0.0, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.2],
    "<user>": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "😒": [0.0, 0.3, 0.2, 0.9, 0.0, 0.0, 0.0, 0.0],
    "😑": [0.0, 0.25, 0.2, 0.85, 0.0, 0.0, 0.0, 0.0],
    "😭": [0.0, 0.6, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1],
    "😎": [0.3, 0.0, 0.1, 0.7, 0.0, 0.1, 0.0, 0.0],
    "😃": [0.6, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0],
}


def _hash_vector(token: str, dim: int = TOY_DIM) -> list[float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values: list[float] = []
    for i in range(dim):
        raw = digest[i] / 255.0
        values.append((raw * 2.0 - 1.0) * 0.15)
    return values


def embed_token(token: str, dim: int = TOY_DIM) -> list[float]:
    if token in SEED_VECTORS:
        vec = list(SEED_VECTORS[token])
        if len(vec) != dim:
            raise ValueError("SEED_VECTORS width does not match dim")
        return vec
    return _hash_vector(token, dim=dim)


def embed_tokens(tokens: Sequence[str], dim: int = TOY_DIM) -> list[list[float]]:
    return [embed_token(token, dim=dim) for token in tokens]


def average_pool(vectors: Sequence[Sequence[float]], dim: int = TOY_DIM) -> list[float]:
    if not vectors:
        return [0.0] * dim
    width = len(vectors[0])
    out = [0.0] * width
    for vec in vectors:
        for idx, value in enumerate(vec):
            out[idx] += value
    return [value / len(vectors) for value in out]


def embed_texts(texts: Iterable[Sequence[str]], dim: int = TOY_DIM) -> list[list[float]]:
    return [average_pool(embed_tokens(tokens, dim=dim), dim=dim) for tokens in texts]
