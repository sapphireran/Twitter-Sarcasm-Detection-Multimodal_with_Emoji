"""Mean-pooling helpers that mirror ``data_utils`` embedding lookup.

``AverageVectorPerTweet`` and ``AverageVectorPerEmoji`` are the same
algorithm with different vocabularies:

1. For each token, if it is in the vector table, keep that row.
2. If the tweet produced at least one hit, return the mean of those rows.
3. Otherwise return a zero vector of the table's width.

The deep model does something different. ``Preprocess`` builds a
``(vocab_size, 200)`` matrix: GloVe if the token is in GloVe, else the
mean of any emoji2vec hits inside the token, else zeros. The examples
expose both paths with tiny in-memory tables so they run without
``glove.twitter.27B.200d.bin``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

VectorTable = Mapping[str, np.ndarray]


def lookup(token: str, table: VectorTable) -> np.ndarray | None:
    """Return the vector for ``token``, or None on a miss.

    Gensim 3 exposed membership as ``token in keyed_vectors.vocab``.
    Gensim 4 removed ``.vocab``; membership is ``token in keyed_vectors``.
    Example code always uses a plain ``dict`` so either era can map onto it.
    """
    vector = table.get(token)
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float64)


def average_vector_per_sequence(
    tokens: Sequence[str],
    table: VectorTable,
    dim: int | None = None,
) -> np.ndarray:
    """Mean-pool one token sequence against ``table``.

    This is the per-tweet body of both ``AverageVectorPerTweet`` and
    ``AverageVectorPerEmoji``.
    """
    if dim is None:
        dim = _infer_dim(table)
    rows = [vec for token in tokens if (vec := lookup(token, table)) is not None]
    if not rows:
        return np.zeros((dim,), dtype=np.float64)
    return np.mean(np.stack(rows, axis=0), axis=0)


def average_corpus(
    docs: Sequence[Sequence[str]],
    table: VectorTable,
    dim: int | None = None,
) -> np.ndarray:
    """Stack ``average_vector_per_sequence`` for every document."""
    if dim is None:
        dim = _infer_dim(table)
    return np.stack(
        [average_vector_per_sequence(doc, table, dim=dim) for doc in docs],
        axis=0,
    )


def multimodal_concat(
    word_vectors: np.ndarray,
    emoji_vectors: np.ndarray,
) -> np.ndarray:
    """Concatenate word and emoji means the way ``ml_read_data`` does.

    Training notebooks feed SVM / trees a 200-d GloVe mean (single-modal)
    or a 400-d ``[word | emoji]`` mean (multi-modal).
    """
    word_vectors = np.asarray(word_vectors)
    emoji_vectors = np.asarray(emoji_vectors)
    if word_vectors.shape[0] != emoji_vectors.shape[0]:
        raise ValueError(
            "word and emoji batches must have the same length, "
            f"got {word_vectors.shape[0]} and {emoji_vectors.shape[0]}"
        )
    return np.concatenate([word_vectors, emoji_vectors], axis=-1)


def build_embedding_matrix(
    word_index: Mapping[str, int],
    vocab_size: int,
    word_table: VectorTable,
    emoji_table: VectorTable | None = None,
    dim: int = 200,
    use_emoji_fallback: bool = True,
) -> np.ndarray:
    """Build the frozen embedding matrix used by ``Preprocess``.

    Parameters
    ----------
    word_index:
        Tokenizer mapping ``token -> integer id``. Index 0 is padding.
    vocab_size:
        Number of rows in the matrix. The 2023 code passes ``len(docs)``
        (``count`` from ``ReadOpen``), which is *not* ``len(word_index)+1``.
        That is a real quirk of the original notebook; examples let you
        pass a correct size when you want one.
    word_table:
        GloVe-like table.
    emoji_table:
        emoji2vec-like table used only when the token misses ``word_table``.
    use_emoji_fallback:
        ``Preprocess(..., get_emoji2vec=True)`` for the multi-modal net,
        ``False`` for the single-modal net (emoji rows become zeros).
    """
    matrix = np.zeros((vocab_size, dim), dtype=np.float64)
    emoji_table = emoji_table or {}
    for word, index in word_index.items():
        if index >= vocab_size:
            continue
        word_vec = lookup(word, word_table)
        if word_vec is not None:
            matrix[index] = word_vec[:dim]
            continue
        if not use_emoji_fallback:
            continue
        emoji_rows = [
            vec
            for token in _chars_and_self(word)
            if (vec := lookup(token, emoji_table)) is not None
        ]
        if emoji_rows:
            matrix[index] = np.mean(np.stack(emoji_rows), axis=0)[:dim]
    return matrix


def _chars_and_self(word: str) -> list[str]:
    """Yield the token itself plus each character (emoji fallback scan)."""
    return [word, *list(word)]


def _infer_dim(table: VectorTable) -> int:
    for vector in table.values():
        return int(np.asarray(vector).shape[-1])
    raise ValueError("cannot infer embedding width from an empty table")
