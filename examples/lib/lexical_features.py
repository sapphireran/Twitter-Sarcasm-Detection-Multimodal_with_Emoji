"""Surface features for the lexical sarcasm baseline.

These are the cues called out in docs/annotated_examples.md: sarcasm
hashtags, `#not`, elongation, emoji counts, and a cheap “positive
word plus complaint” flag. They are deliberately not embeddings.
"""

from __future__ import annotations

import re

import numpy as np

from .tweet_tokenize import extract_emoji, extract_hashtags, is_elongated, tokenize_tweet

POSITIVE = frozenset(
    {
        "love",
        "loovee",
        "loved",
        "great",
        "best",
        "awesome",
        "wonderful",
        "glad",
        "happy",
        "yay",
        "lucky",
        "honored",
        "good",
    }
)
COMPLAINT = frozenset(
    {
        "hate",
        "dirty",
        "late",
        "pissed",
        "pain",
        "worst",
        "sick",
        "bored",
        "broken",
        "tired",
        "grungy",
        "useless",
        "annoyed",
        "exhausted",
    }
)
SARC_TAGS = frozenset(
    {
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#sarcastictweets",
        "#yeahright",
    }
)

FEATURE_NAMES = (
    "bias",
    "has_not_hashtag",
    "has_sarcasm_hashtag",
    "has_yeahright",
    "hashtag_count",
    "emoji_count",
    "elongated_count",
    "excl_count",
    "quest_count",
    "ellipsis",
    "positive_count",
    "complaint_count",
    "pos_and_complaint",
    "token_count_div20",
)


def lexical_vector(sentence: str) -> np.ndarray:
    tokens = tokenize_tweet(sentence)
    tags = extract_hashtags(tokens)
    emoji = extract_emoji(tokens)
    words = [tok for tok in tokens if tok.isalpha()]

    has_not = float(any(tag == "#not" for tag in tags))
    has_sarc = float(any(tag in SARC_TAGS for tag in tags))
    has_yeah = float(any(tag == "#yeahright" for tag in tags))
    elong = float(sum(1 for tok in tokens if is_elongated(tok)))
    excl = float(sentence.count("!"))
    quest = float(sentence.count("?"))
    ellipsis = float(1.0 if ("..." in sentence or "…" in sentence) else 0.0)
    pos = float(sum(1 for tok in words if tok in POSITIVE))
    comp = float(sum(1 for tok in words if tok in COMPLAINT))
    # Also count hashtag forms like #loveit as a weak positive.
    pos += float(sum(1 for tag in tags if tag.lstrip("#") in POSITIVE))
    vec = np.array(
        [
            1.0,
            has_not,
            has_sarc,
            has_yeah,
            float(len(tags)),
            float(len(emoji)),
            elong,
            excl,
            quest,
            ellipsis,
            pos,
            comp,
            float(pos > 0 and comp > 0),
            float(len(tokens)) / 20.0,
        ],
        dtype=np.float64,
    )
    if vec.shape != (len(FEATURE_NAMES),):
        raise RuntimeError("FEATURE_NAMES / lexical_vector length mismatch")
    return vec


def lexical_feature_matrix(sentences: list[str]) -> np.ndarray:
    if not sentences:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.stack([lexical_vector(s) for s in sentences], axis=0)


def hashtag_rule_predict(sentence: str) -> int:
    """Predict sarcastic iff a sarcasm / #not / #yeahright tag is present."""
    tags = extract_hashtags(tokenize_tweet(sentence))
    if any(tag == "#not" or tag in SARC_TAGS for tag in tags):
        return 1
    return 0


_WORD_RE = re.compile(r"[A-Za-z]+")


def content_words(sentence: str) -> list[str]:
    return _WORD_RE.findall(sentence.lower())
