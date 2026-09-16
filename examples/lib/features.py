"""Hand-built tweet features for the leakage-aware heuristic baseline.

The feature names are stable: tests and the Naive Bayes example both depend on
the dict key order from ``FEATURE_NAMES``.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from .tokenize import tokenize_tweet

# Distant-supervision tags that often *are* the label.
HASHTAG_LEAK_PATTERNS = (
    "#not",
    "#sarcasm",
    "#sarcastic",
    "#sarcastictweet",
    "#yeahright",
)

_LEAK_RE = re.compile(
    r"#not\b|#sarcasm\b|#sarcastic\b|#sarcastictweet\b|#yeahright\b",
    re.IGNORECASE,
)

# Surface cues that show up on sarcastic lines even after hashtags are ignored.
_POS_FLIP_RE = re.compile(
    r"\b(love|loved|great|awesome|perfect|yay|can't wait|cant wait|so excited)\b",
    re.IGNORECASE,
)
_NEG_SITUATION_RE = re.compile(
    r"\b(wait|waiting|late|work|shift|exam|homework|traffic|monday|rain|broke)\b",
    re.IGNORECASE,
)

SARCASTIC_LEANING_EMOJI = frozenset("😒😑🙄😏😩😤")
SINCERE_LEANING_EMOJI = frozenset("😍❤😘😊♥")

FEATURE_NAMES: Sequence[str] = (
    "has_leak_hashtag",
    "has_pos_flip_word",
    "has_neg_situation",
    "has_sarcastic_emoji",
    "has_sincere_emoji",
    "has_any_emoji",
    "has_user_mention",
    "is_short",
    "is_long",
    "has_ellipsis",
    "has_exclaim",
    "has_question",
)


def has_leak_hashtag(text: str) -> bool:
    return bool(_LEAK_RE.search(text))


def extract_features(text: str, *, include_leak: bool = True) -> Dict[str, int]:
    """Return a 0/1 feature dict.

    When ``include_leak`` is false, ``has_leak_hashtag`` is forced to 0 so a
    second model can be trained on the same tweets without the meta-label.
    """
    tokens = tokenize_tweet(text)
    token_set = set(tokens)
    joined = " ".join(tokens)
    emoji_chars = [tok for tok in tokens if tok in SARCASTIC_LEANING_EMOJI or tok in SINCERE_LEANING_EMOJI]
    any_emoji = any(len(tok) == 1 and ord(tok) > 255 for tok in tokens) or bool(emoji_chars)
    n = len(tokens)
    feats = {
        "has_leak_hashtag": int(include_leak and has_leak_hashtag(text)),
        "has_pos_flip_word": int(bool(_POS_FLIP_RE.search(text))),
        "has_neg_situation": int(bool(_NEG_SITUATION_RE.search(text))),
        "has_sarcastic_emoji": int(any(tok in SARCASTIC_LEANING_EMOJI for tok in token_set)),
        "has_sincere_emoji": int(any(tok in SINCERE_LEANING_EMOJI for tok in token_set)),
        "has_any_emoji": int(any_emoji),
        "has_user_mention": int("<user>" in token_set or any(tok.startswith("@") for tok in tokens)),
        "is_short": int(n <= 8),
        "is_long": int(n >= 24),
        "has_ellipsis": int("..." in text or "…" in text),
        "has_exclaim": int("!" in text),
        "has_question": int("?" in text),
    }
    # Keep a reference so linters know ``joined`` is intentional documentation
    # of the token stream (useful when debugging a single tweet).
    feats["_token_count"] = n
    feats["_joined"] = 0 if not joined else 0
    # Public vector uses FEATURE_NAMES only.
    return {name: int(feats[name]) for name in FEATURE_NAMES}


def feature_matrix(
    texts: Iterable[str], *, include_leak: bool = True
) -> List[Dict[str, int]]:
    return [extract_features(text, include_leak=include_leak) for text in texts]
