"""Tiny stand-ins for the two 2023 fusion styles.

Classical path
    mean GloVe (200)  and  mean GloVe ⊕ mean emoji2vec (400)

Neural path
    one sequence of 200-d rows; emoji tokens write emoji2vec into the
    *same* table instead of a second channel

The vectors here are synthetic and short so the walkthrough can print
them. Shapes and the zero-vector fallback match ``data_utils``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .dataset_io import is_emoji_char
from .tokenize import tokenize_tweet


@dataclass(frozen=True)
class PooledTweet:
    tokens: List[str]
    word_hits: List[str]
    emoji_hits: List[str]
    word_mean: np.ndarray
    emoji_mean: np.ndarray

    @property
    def concatenated(self) -> np.ndarray:
        return np.concatenate([self.word_mean, self.emoji_mean], axis=0)

    @property
    def word_dim(self) -> int:
        return int(self.word_mean.shape[0])

    @property
    def fused_dim(self) -> int:
        return int(self.concatenated.shape[0])


def _lookup_mean(
    tokens: Sequence[str],
    table: Mapping[str, np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, List[str]]:
    rows = []
    hits: List[str] = []
    for token in tokens:
        if token in table:
            rows.append(np.asarray(table[token], dtype=np.float64))
            hits.append(token)
    if not rows:
        return np.zeros((dim,), dtype=np.float64), hits
    return np.mean(np.stack(rows, axis=0), axis=0), hits


def pool_tweet(
    sentence: str,
    word_table: Mapping[str, np.ndarray],
    emoji_table: Mapping[str, np.ndarray],
    dim: int,
) -> PooledTweet:
    tokens = tokenize_tweet(sentence)
    word_mean, word_hits = _lookup_mean(tokens, word_table, dim)
    emoji_mean, emoji_hits = _lookup_mean(tokens, emoji_table, dim)
    return PooledTweet(
        tokens=tokens,
        word_hits=word_hits,
        emoji_hits=emoji_hits,
        word_mean=word_mean,
        emoji_mean=emoji_mean,
    )


def pool_corpus(
    sentences: Sequence[str],
    word_table: Mapping[str, np.ndarray],
    emoji_table: Mapping[str, np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    single = []
    fused = []
    for sentence in sentences:
        pooled = pool_tweet(sentence, word_table, emoji_table, dim)
        single.append(pooled.word_mean)
        fused.append(pooled.concatenated)
    return np.stack(single, axis=0), np.stack(fused, axis=0)


@dataclass(frozen=True)
class SequenceEmbedding:
    tokens: List[str]
    ids: List[int]
    matrix: np.ndarray
    used_emoji_fallback: List[str]


def build_shared_embedding_matrix(
    token_docs: Sequence[Sequence[str]],
    word_table: Mapping[str, np.ndarray],
    emoji_table: Mapping[str, np.ndarray],
    dim: int,
    use_emoji_fallback: bool,
) -> Tuple[Dict[str, int], np.ndarray, List[str]]:
    """Mirror ``Preprocess`` on a toy vocabulary.

    The 2023 code sized the matrix to *tweet count*. Here the matrix is
    sized to ``vocab + 1`` (index 0 = pad), which is the layout Keras
    actually needs.
    """

    index: Dict[str, int] = {}
    for doc in token_docs:
        for token in doc:
            if token not in index:
                index[token] = len(index) + 1
    matrix = np.zeros((len(index) + 1, dim), dtype=np.float64)
    fallbacks: List[str] = []
    for token, row_id in index.items():
        if token in word_table:
            matrix[row_id] = np.asarray(word_table[token], dtype=np.float64)
            continue
        emoji_chars = [char for char in token if is_emoji_char(char) or char in emoji_table]
        if use_emoji_fallback and emoji_chars:
            rows = [np.asarray(emoji_table[char], dtype=np.float64) for char in emoji_chars if char in emoji_table]
            if rows:
                matrix[row_id] = np.mean(np.stack(rows, axis=0), axis=0)
                fallbacks.append(token)
    return index, matrix, fallbacks


def encode_sequence(
    sentence: str,
    word_index: Mapping[str, int],
    matrix: np.ndarray,
    pad_to: int,
) -> SequenceEmbedding:
    tokens = tokenize_tweet(sentence)
    ids = [word_index[token] for token in tokens if token in word_index]
    padded = ids + [0] * max(pad_to - len(ids), 0)
    padded = padded[:pad_to]
    rows = matrix[np.array(padded, dtype=int)]
    used = [token for token in tokens if token not in word_index]
    return SequenceEmbedding(tokens=tokens, ids=padded, matrix=rows, used_emoji_fallback=used)


def toy_tables(dim: int = 4) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Deterministic mini GloVe / emoji2vec tables for walkthroughs."""

    rng = np.random.default_rng(2023)
    words = {
        "i": rng.normal(size=dim),
        "love": rng.normal(loc=0.8, size=dim),
        "hate": rng.normal(loc=-0.8, size=dim),
        "monday": rng.normal(size=dim),
        "mornings": rng.normal(size=dim),
        "walking": rng.normal(size=dim),
        "to": rng.normal(size=dim),
        "school": rng.normal(size=dim),
        "great": rng.normal(loc=0.6, size=dim),
        "day": rng.normal(size=dim),
        "#not": rng.normal(loc=-1.0, size=dim),
        "#sarcasm": rng.normal(loc=-1.2, size=dim),
    }
    emoji = {
        "😒": rng.normal(loc=-0.9, size=dim),
        "😅": rng.normal(loc=-0.4, size=dim),
        "😂": rng.normal(loc=0.2, size=dim),
        "😭": rng.normal(loc=-0.3, size=dim),
        "👌": rng.normal(loc=0.5, size=dim),
        "😃": rng.normal(loc=0.7, size=dim),
    }
    return words, emoji
