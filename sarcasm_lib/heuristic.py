"""An explicit-cue sarcasm detector for documentation examples.

This is not a replacement for the Bi-LSTM + attention models. It exists so the
docs can show:

* how strong surface cues (#not, #sarcastictweet) are in this corpus
* why the emoji-only subtest looks easier than the full test set
* a baseline that runs without GloVe, emoji2vec, or TensorFlow

The scoring rules are deliberately simple and documented so a reader can
disagree with them and still reproduce the numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .emoji import NEGATIVE_EMOJI, POSITIVE_EMOJI, extract_emojis
from .tokenize import hashtags, tokenize_tweet

# Hashtags that almost always mark sarcasm in the course splits. Counts on
# train: #not 3364/127, #sarcastictweet 228/1, #yeahright 237/4, #sarcastic 283/4.
SARCASM_HASHTAGS = frozenset(
    {
        "#not",
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#yeahright",
        "#notreally",
        "#asif",
        "#ohwait",
        "#notcool",
        "#justgoaway",
        "#iknowido",
    }
)

# Positive openers that often collide with a complaint in this dataset.
_POSITIVE_OPENER_RE = re.compile(
    r"\b(i (just )?love|don't you (just )?love|nothing like|good thing|"
    r"best (feeling|day)|so (fun|excited|happy)|yay)\b",
    re.IGNORECASE,
)

_POSITIVE_HASHTAGS = frozenset(
    {"#love", "#yay", "#great", "#cantwait", "#bestfeelingever", "#loveit"}
)


@dataclass(frozen=True)
class HeuristicResult:
    """Scored prediction plus the cues that fired."""

    text: str
    score: float
    predicted: int
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label_name(self) -> str:
        return "sarcastic" if self.predicted == 1 else "non-sarcastic"


def score_tweet(text: str, *, threshold: float = 1.0) -> HeuristicResult:
    """Score one tweet with a small set of explicit cues.

    A tweet is predicted sarcastic when ``score >= threshold``. Marker
    hashtags contribute 2.0 so they always clear the default threshold.
    """
    tokens = tokenize_tweet(text)
    tags = {tag.lower() for tag in hashtags(tokens)}
    emojis = extract_emojis(text)
    reasons: list[str] = []
    score = 0.0

    marker_hits = sorted(tags & SARCASM_HASHTAGS)
    if marker_hits:
        score += 2.0
        reasons.append("sarcasm hashtag: " + ", ".join(marker_hits))

    opener = _POSITIVE_OPENER_RE.search(text)
    negative_emoji = [glyph for glyph in emojis if glyph in NEGATIVE_EMOJI]
    positive_emoji = [glyph for glyph in emojis if glyph in POSITIVE_EMOJI]

    if opener and negative_emoji:
        score += 1.0
        reasons.append(
            f"positive opener {opener.group(0)!r} with negative emoji "
            + " ".join(negative_emoji)
        )

    if opener and (tags & (SARCASM_HASHTAGS | {"#fml", "#exhausted"})):
        if "positive opener" not in " ".join(reasons):
            score += 0.5
            reasons.append(f"positive opener {opener.group(0)!r} plus cue hashtag")

    # Contrast: cheerful hashtag stacked on a complaint emoji.
    if (tags & _POSITIVE_HASHTAGS) and negative_emoji:
        score += 0.75
        reasons.append("positive hashtag with negative emoji")

    # Bare polarity clash without an explicit #not still happens on the full test.
    if opener and not marker_hits and negative_emoji:
        # already scored above; keep the branch for readability
        pass

    if positive_emoji and negative_emoji and not marker_hits:
        score += 0.25
        reasons.append("mixed-valence emoji without an explicit marker")

    predicted = 1 if score >= threshold else 0
    return HeuristicResult(
        text=text,
        score=score,
        predicted=predicted,
        reasons=tuple(reasons),
    )


def predict_sarcasm(text: str, *, threshold: float = 1.0) -> int:
    return score_tweet(text, threshold=threshold).predicted


def predict_many(texts: list[str], *, threshold: float = 1.0) -> list[int]:
    return [predict_sarcasm(text, threshold=threshold) for text in texts]
