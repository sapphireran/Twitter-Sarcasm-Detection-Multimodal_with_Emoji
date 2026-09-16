"""Tweet-oriented tokenizer used by the personal examples.

The 2023 pipeline called ``nltk.tokenize.TweetTokenizer``. That dependency
is not required for the docs/examples, so this module keeps hashtags,
@-mentions, URLs, and emoji clusters as single tokens and lowercases
everything — close enough to inspect cue leakage and build bag-of-cue
features.

It is *not* a drop-in replacement for the notebook tokenizer. The
reproduction guide still points at NLTK for the original experiments.
"""

from __future__ import annotations

import re
from typing import Iterable

# Hashtags / cashtags first so "#not" stays one token.
# Mentions keep the leading @. URLs are coarse on purpose.
_TOKEN_RE = re.compile(
    r"(https?://\S+|www\.\S+)"
    r"|(\$[\w]+)"
    r"|(#\w+)"
    r"|(@\w+)"
    r"|(<user>)"
    r"|([\U0001F300-\U0001F64F\U0001F680-\U0001F6FF"
    r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF"
    r"\u2600-\u26FF\u2700-\u27BF]+)"
    r"|([A-Za-z]+(?:'[A-Za-z]+)?)"
    r"|(\d+(?:\.\d+)?)"
    r"|([^\s\w])",
    re.UNICODE,
)

_ELONGATION_RE = re.compile(r"(.)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")


def tokenize_tweet(text: str, lowercase: bool = True) -> list[str]:
    """Split a tweet into tokens without NLTK.

    >>> tokenize_tweet("I just love Mondays #not 😒")
    ['i', 'just', 'love', 'mondays', '#not', '😒']
    """
    if not text:
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if lowercase:
            token = token.lower()
        tokens.append(token)
    return tokens


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def has_elongation(text: str) -> bool:
    """True when a character repeats 3+ times (``loovee``, ``yayyy``)."""
    return bool(_ELONGATION_RE.search(text.lower()))


def hashtags(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token.startswith("#")]


def mentions(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token.startswith("@") or token == "<user>"]


def emoji_tokens(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if _is_emoji_token(token)]


def _is_emoji_token(token: str) -> bool:
    if not token:
        return False
    code = ord(token[0])
    return (
        0x1F300 <= code <= 0x1F64F
        or 0x1F680 <= code <= 0x1F6FF
        or 0x1F900 <= code <= 0x1F9FF
        or 0x1FA70 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x26FF
        or 0x2700 <= code <= 0x27BF
    )
