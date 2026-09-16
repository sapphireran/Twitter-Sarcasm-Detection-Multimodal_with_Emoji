"""Lexical sarcasm cues and per-tweet profiles.

The official test split is cue-heavier than train. ``#not`` alone covers
about 26% of test tweets versus 9% of train, and almost every hit is
labeled sarcastic. That leakage is why a hashtag rule is a strong
classroom baseline and why every 2023 model looked better on subtest.

``#not ready yet`` is the noisy case: the token is ``#not``, but the
tweet is often a calendar complaint rather than sarcasm. Profiles keep
the raw token so later analysis can split those cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ccs2lab.tokenize import hashtags, tokenize_tweet

# Exact hashtag tokens after lowercasing. Do not prefix-match: #notes
# and #nothing are not sarcasm markers.
NOT_CUES = frozenset({"#not", "#notreally"})

SARCASM_TAG_CUES = frozenset(
    {
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#sarcasticcomment",
        "#irony",
        "#ironic",
        "#yeahright",
        "#yeahok",
        "#asif",
    }
)

EXPLICIT_CUES = NOT_CUES | SARCASM_TAG_CUES

# Surface words that often sit next to a sarcastic hashtag. Used as a
# weak contrast flag, not as a standalone label.
POSITIVE_WORDS = frozenset(
    {
        "love",
        "loved",
        "loves",
        "great",
        "greatest",
        "awesome",
        "amazing",
        "wonderful",
        "perfect",
        "best",
        "happy",
        "glad",
        "yay",
        "yayy",
        "excited",
        "thrilled",
        "fantastic",
        "beautiful",
    }
)

NEGATIVE_EMOJI = frozenset(
    {
        "😒",
        "😑",
        "🙄",
        "😠",
        "😡",
        "💀",
        "🔫",
        "😭",
        "😩",
        "😔",
        "😞",
        "😕",
        "🙃",
    }
)

# Letter elongation such as "loovee" / "sooo". Dots are not letters.
_ELONG_RE_SOURCE = r"([a-z])\1{2,}"


@dataclass(frozen=True)
class CueProfile:
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    explicit: tuple[str, ...]
    has_not_tag: bool
    has_sarcasm_tag: bool
    has_explicit: bool
    has_emoji: bool
    has_positive: bool
    has_negative_emoji: bool
    has_contrast: bool
    has_elongation: bool
    has_question: bool
    has_exclaim: bool
    has_user: bool

    def cue_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.has_not_tag:
            names.append("not_tag")
        if self.has_sarcasm_tag:
            names.append("sarcasm_tag")
        if self.has_contrast:
            names.append("contrast")
        if self.has_elongation:
            names.append("elongation")
        return tuple(names)


@dataclass
class CueCounts:
    n: int = 0
    sarcastic: int = 0
    hits: int = 0
    hits_sarcastic: int = 0

    def add(self, hit: bool, label: int) -> None:
        self.n += 1
        self.sarcastic += int(label == 1)
        if hit:
            self.hits += 1
            self.hits_sarcastic += int(label == 1)

    @property
    def coverage(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def hit_precision(self) -> float:
        return self.hits_sarcastic / self.hits if self.hits else 0.0


def _has_emoji_token(tokens: Iterable[str]) -> bool:
    for token in tokens:
        if any(ord(char) > 0x2100 for char in token) and not token.startswith(("#", "@", "<")):
            # Cheap filter: keep only tokens that look like symbol/emoji
            # rather than latin words. Hashtags already excluded.
            if any(ord(char) >= 0x2190 for char in token):
                return True
    return False


def profile_tokens(tokens: list[str]) -> CueProfile:
    tags = tuple(hashtags(tokens))
    explicit = tuple(tag for tag in tags if tag in EXPLICIT_CUES)
    has_not_tag = any(tag in NOT_CUES for tag in tags)
    has_sarcasm_tag = any(tag in SARCASM_TAG_CUES for tag in tags)
    has_positive = any(token in POSITIVE_WORDS for token in tokens)
    has_emoji = _has_emoji_token(tokens)
    has_negative_emoji = any(token in NEGATIVE_EMOJI for token in tokens)
    # Contrast is positive wording plus a negative emoji *or* an explicit
    # cue. Sincere "love you 😭" without a sarcasm tag is not contrast.
    has_contrast = has_positive and (has_negative_emoji or bool(explicit))
    has_elongation = any(
        len(token) >= 4 and any(token[i] == token[i + 1] == token[i + 2] and token[i].isalpha() for i in range(len(token) - 2))
        for token in tokens
        if token[:1].isalpha()
    )
    return CueProfile(
        tokens=tuple(tokens),
        tags=tags,
        explicit=explicit,
        has_not_tag=has_not_tag,
        has_sarcasm_tag=has_sarcasm_tag,
        has_explicit=bool(explicit),
        has_emoji=has_emoji,
        has_positive=has_positive,
        has_negative_emoji=has_negative_emoji,
        has_contrast=has_contrast,
        has_elongation=has_elongation,
        has_question="?" in tokens or any(token.startswith("?") for token in tokens),
        has_exclaim="!" in tokens or any(token.startswith("!") for token in tokens),
        has_user="<user>" in tokens or any(token.startswith("@") for token in tokens),
    )


def profile_text(text: str) -> CueProfile:
    return profile_tokens(tokenize_tweet(text))


def cue_rule_label(profile: CueProfile) -> int:
    """Predict sarcastic iff an explicit sarcasm hashtag is present."""
    return int(profile.has_explicit)


def collect_counts(texts: Iterable[str], labels: Iterable[int]) -> dict[str, CueCounts]:
    keys = (
        "explicit",
        "not_tag",
        "sarcasm_tag",
        "emoji",
        "contrast",
        "elongation",
        "question",
        "exclaim",
    )
    counts = {key: CueCounts() for key in keys}
    for text, label in zip(texts, labels, strict=True):
        profile = profile_text(text)
        flags = {
            "explicit": profile.has_explicit,
            "not_tag": profile.has_not_tag,
            "sarcasm_tag": profile.has_sarcasm_tag,
            "emoji": profile.has_emoji,
            "contrast": profile.has_contrast,
            "elongation": profile.has_elongation,
            "question": profile.has_question,
            "exclaim": profile.has_exclaim,
        }
        for key, hit in flags.items():
            counts[key].add(hit, label)
    return counts
