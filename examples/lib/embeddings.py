"""Tiny embedding-table helpers that mirror ``AverageVectorPerTweet``.

The real project uses 200-d GloVe. These functions use a hand-written toy table
so ``examples/embedding_average_demo.py`` can show the *same averaging rule*
without a 1 GB download.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from .tokenize import tokenize_tweet

Vector = List[float]


def toy_table(dim: int = 8) -> Dict[str, Vector]:
    """A tiny deterministic embedding table for walkthroughs.

    Directions are chosen so that polarity words, sarcasm hashtags, and emoji
    live in distinguishable regions. They are not trained.
    """
    if dim < 4:
        raise ValueError("dim must be at least 4")

    def vec(*values: float) -> Vector:
        out = list(values)
        if len(out) < dim:
            out.extend([0.0] * (dim - len(out)))
        return out[:dim]

    table = {
        "love": vec(1.0, 0.2, 0.0, 0.0),
        "great": vec(0.9, 0.3, 0.0, 0.0),
        "awesome": vec(0.8, 0.4, 0.0, 0.0),
        "hate": vec(-1.0, 0.2, 0.0, 0.0),
        "terrible": vec(-0.9, 0.3, 0.0, 0.0),
        "wait": vec(0.0, 0.0, 0.7, 0.0),
        "waiting": vec(0.0, 0.0, 0.8, 0.0),
        "late": vec(0.0, 0.0, 0.6, 0.1),
        "work": vec(0.0, 0.0, 0.5, 0.0),
        "#not": vec(0.0, 0.0, 0.0, 1.0),
        "#sarcasm": vec(0.0, 0.0, 0.0, 0.9),
        "#yeahright": vec(0.0, 0.0, 0.0, 0.8),
        "😑": vec(0.1, 0.0, 0.2, 0.7),
        "😒": vec(0.1, 0.0, 0.1, 0.6),
        "😍": vec(0.7, 0.4, 0.0, 0.0),
        "😂": vec(0.2, 0.6, 0.0, 0.2),
        "i": vec(0.0, 0.1, 0.0, 0.0),
        "just": vec(0.0, 0.1, 0.0, 0.0),
        "having": vec(0.0, 0.0, 0.1, 0.0),
        "when": vec(0.0, 0.0, 0.1, 0.0),
        "people": vec(0.1, 0.0, 0.0, 0.0),
    }
    return table


def average_vectors(
    tokens: Sequence[str],
    table: Mapping[str, Sequence[float]],
    dim: int,
) -> Vector:
    """Mean of in-vocabulary tokens; zeros if nothing matched.

    This is the rule in ``data_utils.AverageVectorPerTweet`` / ``AverageVectorPerEmoji``.
    """
    rows: List[Sequence[float]] = []
    for tok in tokens:
        if tok in table:
            rows.append(table[tok])
    if not rows:
        return [0.0] * dim
    acc = [0.0] * dim
    for row in rows:
        if len(row) != dim:
            raise ValueError(f"embedding dim {len(row)} != expected {dim}")
        for i, value in enumerate(row):
            acc[i] += value
    n = float(len(rows))
    return [value / n for value in acc]


def embed_tweet(
    text: str,
    table: Mapping[str, Sequence[float]],
    dim: int,
    *,
    emoji_table: Mapping[str, Sequence[float]] | None = None,
) -> Vector:
    """Single-modal mean, or concat(word_mean, emoji_mean) when ``emoji_table`` is set."""
    tokens = tokenize_tweet(text)
    word_vec = average_vectors(tokens, table, dim)
    if emoji_table is None:
        return word_vec
    emoji_vec = average_vectors(tokens, emoji_table, dim)
    return list(word_vec) + list(emoji_vec)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def nearest(
    query: Sequence[float],
    labels: Iterable[str],
    table: Mapping[str, Sequence[float]],
    k: int = 3,
) -> List[tuple[str, float]]:
    scored = [(name, cosine(query, table[name])) for name in labels if name in table]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
