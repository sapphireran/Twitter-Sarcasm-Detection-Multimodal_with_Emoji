"""Surface sarcasm cues: hashtags, emoji, and a small phrase lexicon.

These features are the kind of signal discussed in Davidov et al. (2010)
and Riloff et al. (2013). They are *not* the GloVe / emoji2vec channels
used by the 2023 neural model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokenize import is_emoji_token, is_hashtag

# Hashtags that were commonly used as distant-supervision labels on Twitter.
SARCASM_HASHTAGS = frozenset(
    {
        "#not",
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#yeahright",
        "#shocker",
        "#obviously",
        "#notreally",
        "#kidding",
        "#jk",
        "#asif",
        "#surejan",
    }
)

# Positive openers that often precede a reversal ("I love walking to school").
POSITIVE_STEMS = (
    "i love",
    "i loovee",
    "just love",
    "love when",
    "love walking",
    "good thing",
    "feeling like a million",
    "can't wait",
    "cant wait",
    "so happy",
    "best day",
    "great time",
    "yay",
)

NEGATIVE_EMOTION_EMOJI = frozenset("😒😑😩😅😔😭🔫👎💔🙄😡☹️🙁😕😤😒")


@dataclass(frozen=True)
class CueFeatures:
    n_tokens: int
    n_hashtags: int
    n_emoji: int
    n_users: int
    n_urls: int
    n_ellipsis: int
    has_sarcasm_hashtag: bool
    has_negative_emoji: bool
    has_positive_stem: bool
    sarcasm_hashtags: tuple[str, ...] = field(default_factory=tuple)
    emoji: tuple[str, ...] = field(default_factory=tuple)
    hashtags: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, float]:
        return {
            "n_tokens": float(self.n_tokens),
            "n_hashtags": float(self.n_hashtags),
            "n_emoji": float(self.n_emoji),
            "n_users": float(self.n_users),
            "n_urls": float(self.n_urls),
            "n_ellipsis": float(self.n_ellipsis),
            "has_sarcasm_hashtag": float(self.has_sarcasm_hashtag),
            "has_negative_emoji": float(self.has_negative_emoji),
            "has_positive_stem": float(self.has_positive_stem),
        }

    def dense_vector(self) -> list[float]:
        d = self.as_dict()
        return [d[key] for key in sorted(d)]


def extract_cues(text: str, tokens: list[str]) -> CueFeatures:
    hashtags = tuple(tok for tok in tokens if is_hashtag(tok))
    emoji = tuple(tok for tok in tokens if is_emoji_token(tok))
    lowered = text.lower()
    sarcasm_tags = tuple(tag for tag in hashtags if tag in SARCASM_HASHTAGS)
    return CueFeatures(
        n_tokens=len(tokens),
        n_hashtags=len(hashtags),
        n_emoji=len(emoji),
        n_users=sum(1 for tok in tokens if tok == "<user>"),
        n_urls=sum(1 for tok in tokens if tok == "<url>"),
        n_ellipsis=sum(1 for tok in tokens if tok == "..." or tok == "…"),
        has_sarcasm_hashtag=bool(sarcasm_tags),
        has_negative_emoji=any(ch in NEGATIVE_EMOTION_EMOJI for tok in emoji for ch in tok),
        has_positive_stem=any(stem in lowered for stem in POSITIVE_STEMS),
        sarcasm_hashtags=sarcasm_tags,
        emoji=emoji,
        hashtags=hashtags,
    )
