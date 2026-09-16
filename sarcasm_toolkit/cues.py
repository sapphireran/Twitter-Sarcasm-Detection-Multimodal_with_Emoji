"""Lexical sarcasm cues observed in this personal course dataset.

Hashtag leakage is a first-class property of the SemEval-style Twitter
sarcasm splits used here. On the held-out test set, about 64% of
sarcastic tweets contain an explicit cue hashtag (``#not``, ``#sarcasm``,
``#sarcastictweet``, …) while almost none of the literal tweets do.
The emoji-only *subtest* is even more extreme.

The original BiLSTM still beats a hashtag lexicon — that is the point of
the 2023 project — but examples should show the cue baseline first so
the embedding results have a floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .tokenize import emoji_tokens, has_elongation, hashtags, tokenize_tweet

# Explicit self-annotation used as distant supervision in many Twitter
# sarcasm papers. Values are lowercase including the leading '#'.
CUE_HASHTAGS = frozenset(
    {
        "#not",
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#irony",
        "#ironic",
        "#yeahright",
    }
)

POSITIVE_VALENCE = frozenset(
    {
        "love",
        "loved",
        "loves",
        "great",
        "greatest",
        "best",
        "awesome",
        "perfect",
        "wonderful",
        "yay",
        "yayy",
        "happy",
        "glad",
        "excited",
        "thrilled",
    }
)

CONTRAST_FRAMES = (
    "love when",
    "love it when",
    "just love",
    "so happy",
    "best feeling",
    "great to",
)


@dataclass(frozen=True)
class CueFeatures:
    """Named cue vector for one tweet."""

    values: tuple[float, ...]

    def as_list(self) -> list[float]:
        return list(self.values)


def feature_names() -> tuple[str, ...]:
    return (
        "has_not_hashtag",
        "has_sarcasm_family",
        "has_yeahright",
        "has_irony_family",
        "any_cue_hashtag",
        "cue_hashtag_count",
        "hashtag_count",
        "emoji_count",
        "has_emoji",
        "has_user",
        "has_elongation",
        "has_ellipsis",
        "has_exclaim",
        "positive_valence",
        "contrast_frame",
        "token_count",
        "char_count",
    )


SARCASM_FAMILY = frozenset({"#sarcasm", "#sarcastic", "#sarcastictweet"})
IRONY_FAMILY = frozenset({"#irony", "#ironic"})


def extract_cue_features(text: str, tokens: Sequence[str] | None = None) -> CueFeatures:
    """Turn a tweet into the fixed-length cue vector used by examples."""
    lowered = text.lower()
    toks = list(tokens) if tokens is not None else tokenize_tweet(text)
    tag_set = {tag.lower() for tag in hashtags(toks)}
    emojis = emoji_tokens(toks)

    has_not = 1.0 if "#not" in tag_set else 0.0
    has_sarc = 1.0 if tag_set & SARCASM_FAMILY else 0.0
    has_yeah = 1.0 if "#yeahright" in tag_set else 0.0
    has_irony = 1.0 if tag_set & IRONY_FAMILY else 0.0
    cue_tags = tag_set & CUE_HASHTAGS

    values = (
        has_not,
        has_sarc,
        has_yeah,
        has_irony,
        1.0 if cue_tags else 0.0,
        float(len(cue_tags)),
        float(len(tag_set)),
        float(len(emojis)),
        1.0 if emojis else 0.0,
        1.0 if ("<user>" in toks or any(t.startswith("@") for t in toks)) else 0.0,
        1.0 if has_elongation(text) else 0.0,
        1.0 if "..." in text or "…" in text else 0.0,
        1.0 if "!" in text else 0.0,
        1.0 if any(tok in POSITIVE_VALENCE for tok in toks) else 0.0,
        1.0 if any(frame in lowered for frame in CONTRAST_FRAMES) else 0.0,
        float(len(toks)),
        float(len(text)),
    )
    return CueFeatures(values=values)


def extract_matrix(texts: Iterable[str]) -> list[list[float]]:
    return [extract_cue_features(text).as_list() for text in texts]


def cue_hashtags_in(text: str) -> list[str]:
    tokens = tokenize_tweet(text)
    return sorted(set(hashtags(tokens)) & CUE_HASHTAGS)
