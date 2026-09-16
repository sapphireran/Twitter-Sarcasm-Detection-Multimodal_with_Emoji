"""Hand-built sarcasm cues.

These are the shortcuts a frozen GloVe table can still exploit: hashtags,
elongation, contrast templates, a few faces. The lexical baseline exists so
the neural numbers have a floor that does not pretend to be semantic.
"""

from __future__ import annotations

import re

import numpy as np

from .emoji_extract import extract_emojis
from .tweet_tokenize import tokenize_tweet

_ELONG = re.compile(r"(.)\1{2,}")
_HASHTAG = re.compile(r"#\w+", re.I)

FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "n_tokens",
    "n_chars",
    "n_hashtags",
    "n_mentions",
    "n_urls",
    "n_emoji",
    "n_excl",
    "n_quest",
    "n_ellipsis",
    "has_not_tag",
    "has_sarcasm_tag",
    "has_sarcastic_tag",
    "has_yeah_right",
    "has_oh_great",
    "has_love_when",
    "has_just_great",
    "has_so_fun",
    "has_as_if",
    "has_sure",
    "has_yay",
    "has_great",
    "has_love",
    "has_elongation",
    "has_face_unamused",
    "has_face_expressionless",
    "has_face_upside_down",
    "has_face_rolling",
    "has_face_sweat",
    "has_face_blank",
    "has_positive_word_plus_neg_tag",
)


_POS_WORDS = ("love", "great", "yay", "awesome", "wonderful", "perfect", "glad")
_NEG_TAGS = ("#not", "#sarcasm", "#sarcastic", "#sarcastictweet")


def _bool(flag: bool) -> float:
    return 1.0 if flag else 0.0


def featurize(text: str) -> np.ndarray:
    tokens = tokenize_tweet(text, lowercase=True)
    lower = text.casefold()
    joined = " ".join(tokens)
    hashtags = [tok for tok in tokens if tok.startswith("#")]
    emojis = extract_emojis(text)

    has_not = any(tok == "#not" for tok in tokens)
    has_sarc = any(tok.startswith("#sarcas") for tok in tokens)
    has_pos = any(tok in _POS_WORDS for tok in tokens)
    has_neg_tag = any(tok in _NEG_TAGS for tok in tokens)

    values = [
        1.0,
        float(len(tokens)),
        float(len(text)),
        float(len(hashtags)),
        float(sum(1 for tok in tokens if tok in {"<user>"} or tok.startswith("@"))),
        float(sum(1 for tok in tokens if tok.startswith("http") or tok.startswith("www."))),
        float(len(emojis)),
        float(text.count("!")),
        float(text.count("?")),
        float(lower.count("...") + lower.count("…")),
        _bool(has_not),
        _bool("#sarcasm" in hashtags),
        _bool("#sarcastic" in hashtags or "#sarcastictweet" in hashtags),
        _bool("yeah right" in joined or "yah right" in joined),
        _bool("oh great" in joined or "oh good" in joined),
        _bool("love when" in joined or "loovee when" in joined or "love it when" in joined),
        _bool("just great" in joined or "just perfect" in joined),
        _bool("so fun" in joined or "so funny" in joined),
        _bool("as if" in joined),
        _bool("sure" in tokens),
        _bool("yay" in tokens),
        _bool("great" in tokens),
        _bool("love" in tokens or "loovee" in tokens),
        _bool(any(_ELONG.search(tok) for tok in tokens) or bool(_ELONG.search(text))),
        _bool("😒" in text),
        _bool("😑" in text),
        _bool("🙃" in text),
        _bool("🙄" in text),
        _bool("😅" in text or "😓" in text),
        _bool("😐" in text or "😶" in text),
        _bool(has_pos and has_neg_tag),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError("feature schema drifted")
    return np.asarray(values, dtype=np.float64)


def featurize_corpus(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.stack([featurize(text) for text in texts], axis=0)


def cue_presence_table(texts: list[str], labels: list[int]) -> dict[str, dict[str, float]]:
    """Per-feature hit rate on sarcastic vs non-sarcastic rows (skip bias / counts)."""
    X = featurize_corpus(texts)
    y = np.asarray(labels, dtype=np.int32)
    out: dict[str, dict[str, float]] = {}
    binary_names = [n for n in FEATURE_NAMES if n.startswith("has_")]
    for name in binary_names:
        col = FEATURE_NAMES.index(name)
        hits = X[:, col] > 0
        out[name] = {
            "all": float(hits.mean()) if len(hits) else 0.0,
            "sarcastic": float(hits[y == 1].mean()) if (y == 1).any() else 0.0,
            "non_sarcastic": float(hits[y == 0].mean()) if (y == 0).any() else 0.0,
        }
    return out
