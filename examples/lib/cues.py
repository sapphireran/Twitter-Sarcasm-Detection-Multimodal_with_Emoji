"""Surface sarcasm cues that the 2023 project treated as signals.

The course dump is full of explicit markers (``#not``, ``#yeahright``,
``#sarcastictweet``) plus milder contrast patterns (positive adjective
plus a groaning emoji). The deep model did *not* use these rules; they
are here as a transparent baseline so the documented BiLSTM numbers have
something cheap to beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokenize import extract_emojis, extract_hashtags, tokenize_tweet

# Tags that are almost definitional in this dump. Empirically, #not is
# sarcastic on ~97% of the training rows that contain it.
SARCASM_HASHTAGS = frozenset(
    {
        "not",
        "sarcasm",
        "sarcastic",
        "sarcastictweet",
        "sarcastictweets",
        "yeahright",
        "surejan",
        "kidding",
        "asif",
        "obviouslynot",
    }
)

# Positive-looking words that often flip under a sarcasm hashtag.
POSITIVE_LEXICON = frozenset(
    {
        "love",
        "loved",
        "great",
        "awesome",
        "wonderful",
        "perfect",
        "best",
        "happy",
        "excited",
        "yay",
        "fun",
        "glad",
        "fantastic",
        "amazing",
        "thrilled",
    }
)

# Face / gesture emoji that frequently co-occur with sarcastic praise.
GROAN_EMOJI = frozenset(
    {
        "😒",
        "😑",
        "🙄",
        "😩",
        "😭",
        "😅",
        "😏",
        "☹️",
        "🙁",
        "😕",
        "😐",
        "💔",
        "👎",
    }
)


@dataclass(frozen=True)
class CueSet:
    """Bag of surface cues extracted from one tweet."""

    hashtags: tuple[str, ...] = ()
    sarcasm_hashtags: tuple[str, ...] = ()
    emojis: tuple[str, ...] = ()
    groan_emojis: tuple[str, ...] = ()
    positive_words: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()

    @property
    def has_explicit_marker(self) -> bool:
        return bool(self.sarcasm_hashtags)

    @property
    def has_contrast(self) -> bool:
        return bool(self.positive_words) and (
            bool(self.groan_emojis) or self.has_explicit_marker
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "hashtags": list(self.hashtags),
            "sarcasm_hashtags": list(self.sarcasm_hashtags),
            "emojis": list(self.emojis),
            "groan_emojis": list(self.groan_emojis),
            "positive_words": list(self.positive_words),
            "has_explicit_marker": self.has_explicit_marker,
            "has_contrast": self.has_contrast,
            "n_tokens": len(self.tokens),
        }


@dataclass
class RuleDecision:
    """Output of the transparent hashtag / contrast rule."""

    label: int
    reason: str
    cues: CueSet = field(default_factory=CueSet)


def extract_cues(text: str) -> CueSet:
    tokens = tuple(tokenize_tweet(text))
    hashtags = tuple(extract_hashtags(text))
    emojis = tuple(extract_emojis(text))
    return CueSet(
        hashtags=hashtags,
        sarcasm_hashtags=tuple(tag for tag in hashtags if tag in SARCASM_HASHTAGS),
        emojis=emojis,
        groan_emojis=tuple(char for char in emojis if char in GROAN_EMOJI),
        positive_words=tuple(
            token for token in tokens if token in POSITIVE_LEXICON
        ),
        tokens=tokens,
    )


def rule_predict(text: str) -> RuleDecision:
    """Predict sarcastic if an explicit marker or contrast pattern fires.

    This is intentionally conservative: most sarcastic tweets in the
    balanced test set do *not* carry ``#not``. The rule is a lower bound
    that makes the 86–87% BiLSTM accuracy look earned rather than magic.
    """

    cues = extract_cues(text)
    if cues.has_explicit_marker:
        return RuleDecision(label=1, reason="explicit_hashtag", cues=cues)
    if cues.has_contrast:
        return RuleDecision(label=1, reason="positive_plus_groan", cues=cues)
    return RuleDecision(label=0, reason="no_surface_marker", cues=cues)
