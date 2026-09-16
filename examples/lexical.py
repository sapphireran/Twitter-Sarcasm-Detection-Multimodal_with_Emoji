"""Surface features that do not need GloVe or emoji2vec.

The 2023 baselines averaged 200-d embeddings. These features exist so a
walkthrough can train a classifier on the real CSVs in a few seconds and
still talk about the same cues the course project cared about: `#not`,
emoji, elongation, mentions, punctuation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .dataset_io import count_emoji
from .tokenize import tokenize_tweet

FEATURE_NAMES: Tuple[str, ...] = (
    "n_tokens",
    "n_chars",
    "n_emoji",
    "n_hashtags",
    "n_mentions",
    "n_exclaim",
    "n_question",
    "n_ellipsis",
    "has_not_hashtag",
    "has_sarcasm_hashtag",
    "has_yeahright",
    "n_allcaps",
    "n_elongated",
    "punct_ratio",
    "avg_token_len",
    "emoji_token_ratio",
)

_ELONGATED_RE = re.compile(r"(.)\1{2,}")
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+|<user>", re.IGNORECASE)


def _is_allcaps(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    return len(letters) >= 2 and all(char.isupper() for char in letters)


def extract_feature_dict(sentence: str) -> Dict[str, float]:
    tokens = tokenize_tweet(sentence, lowercase=False)
    lower_tokens = [token.lower() for token in tokens]
    n_tokens = max(len(tokens), 1)
    n_chars = len(sentence)
    n_emoji = count_emoji(sentence)
    n_hashtags = len(_HASHTAG_RE.findall(sentence))
    n_mentions = len(_MENTION_RE.findall(sentence))
    joined_lower = " ".join(lower_tokens)
    features = {
        "n_tokens": float(len(tokens)),
        "n_chars": float(n_chars),
        "n_emoji": float(n_emoji),
        "n_hashtags": float(n_hashtags),
        "n_mentions": float(n_mentions),
        "n_exclaim": float(sentence.count("!")),
        "n_question": float(sentence.count("?")),
        "n_ellipsis": float(sentence.count("...")),
        "has_not_hashtag": float(any(token in {"#not", "#n0t"} for token in lower_tokens)),
        "has_sarcasm_hashtag": float(
            any(token.startswith("#") and "sarcas" in token for token in lower_tokens)
        ),
        "has_yeahright": float("#yeahright" in joined_lower or "yeah right" in joined_lower),
        "n_allcaps": float(sum(1 for token in tokens if _is_allcaps(token))),
        "n_elongated": float(sum(1 for token in lower_tokens if _ELONGATED_RE.search(token))),
        "punct_ratio": float(sum(1 for char in sentence if char in ".,;:!?") / max(n_chars, 1)),
        "avg_token_len": float(sum(len(token) for token in tokens) / n_tokens),
        "emoji_token_ratio": float(n_emoji / n_tokens),
    }
    return features


def extract_feature_vector(sentence: str) -> np.ndarray:
    values = extract_feature_dict(sentence)
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract_feature_matrix(sentences: Sequence[str]) -> np.ndarray:
    if not sentences:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([extract_feature_vector(sentence) for sentence in sentences])


def standardize(
    train_x: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = [(matrix - mean) / std for matrix in (train_x,) + others]
    return tuple(scaled)


def top_features_by_abs_weight(
    weights: np.ndarray,
    k: int = 8,
    names: Sequence[str] | None = None,
) -> List[Tuple[str, float]]:
    labels = tuple(FEATURE_NAMES if names is None else names)
    if weights.shape[0] != len(labels):
        raise ValueError("weight vector does not match feature names")
    order = np.argsort(np.abs(weights))[::-1]
    return [(labels[i], float(weights[i])) for i in order[:k]]
