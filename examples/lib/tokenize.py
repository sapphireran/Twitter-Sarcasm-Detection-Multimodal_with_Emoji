"""Tweet-ish tokenizer used by the personal examples.

The 2023 pipeline called ``nltk.TweetTokenizer`` and lower-cased every
token. This module is a smaller stand-in: it keeps mentions, hashtags,
URLs, and emoji as atomic tokens and splits ordinary punctuation.
It is *not* a drop-in replacement for NLTK, but it is close enough to
show how the course dump was turned into embedding lookups.
"""

from __future__ import annotations

import re
from typing import Iterable

# Broad but not exhaustive emoji / pictograph ranges. Good enough for the
# 2010s Twitter dump used in the course project.
_EMOJI_RANGES = (
    (0x2190, 0x21FF),
    (0x2300, 0x27BF),
    (0x2B00, 0x2BFF),
    (0x1F000, 0x1F02F),
    (0x1F0A0, 0x1F0FF),
    (0x1F100, 0x1F1FF),
    (0x1F200, 0x1F2FF),
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0xFE00, 0xFE0F),  # variation selectors
)

_TOKEN_RE = re.compile(
    r"("
    r"https?://\S+"
    r"|www\.\S+"
    r"|@\w+"
    r"|#\w+"
    r"|[\U0001F000-\U0001FAFF]"
    r"|[\u2190-\u21FF\u2300-\u27BF\u2B00-\u2BFF]"
    r"|[A-Za-z0-9]+(?:'[A-Za-z]+)?"
    r"|[^\s]"
    r")"
)

_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)
_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def is_emoji(char: str) -> bool:
    """Return True if ``char`` is a single emoji-ish code point."""

    if not char:
        return False
    code = ord(char[0])
    return any(start <= code <= end for start, end in _EMOJI_RANGES)


def extract_emojis(text: str) -> list[str]:
    """Collect emoji-ish characters, skipping variation selectors alone."""

    emojis: list[str] = []
    for char in text:
        code = ord(char)
        if 0xFE00 <= code <= 0xFE0F:
            continue
        if is_emoji(char):
            emojis.append(char)
    return emojis


def tokenize_tweet(text: str, *, lowercase: bool = True) -> list[str]:
    """Approximate the course project's tweet tokenization.

    Empty / whitespace-only strings yield an empty token list, matching
    how ``ReadOpen`` would store a tweet that had nothing left after
    stripping.
    """

    if not text or not text.strip():
        return []
    tokens = [match.group(0) for match in _TOKEN_RE.finditer(text)]
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def extract_hashtags(text: str) -> list[str]:
    """Hashtag bodies without the leading ``#``, lower-cased."""

    return [match.group(1).lower() for match in _HASHTAG_RE.finditer(text)]


def count_mentions(text: str) -> int:
    return len(_MENTION_RE.findall(text))


def has_url(text: str) -> bool:
    return bool(_URL_RE.search(text))


def flatten_tokens(docs: Iterable[Iterable[str]]) -> list[str]:
    flat: list[str] = []
    for doc in docs:
        flat.extend(doc)
    return flat
