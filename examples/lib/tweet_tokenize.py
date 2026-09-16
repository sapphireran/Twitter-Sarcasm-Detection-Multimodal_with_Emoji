"""A small TweetTokenizer stand-in plus the ReadOpen comma step.

The coursework uses `nltk.TweetTokenizer` after replacing commas with
spaces. This module keeps that first step and then splits on a regex
that treats URLs, @mentions, #hashtags, HTML-ish tokens like `<user>`,
and common emoji as single tokens.
"""

from __future__ import annotations

import re

# Keep `<user>` / `<url>` intact; then URLs, hashtags, mentions,
# leftover angle tokens, emoji runs, words (incl. ascii apostrophes),
# and any other non-space glyph.
_TOKEN_RE = re.compile(
    r"<[^>\s]+>"
    r"|https?://\S+"
    r"|www\.\S+"
    r"|#\w+"
    r"|@\w+"
    r"|\.{3}|…"
    r"|[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]"
    r"|[A-Za-z0-9]+(?:'[A-Za-z]+)?"
    r"|[^\s]",
    flags=re.UNICODE,
)

# Repeated letters used as a lengthening cue: loovee, soooo, grrreat.
_ELONG_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)


def strip_commas_like_readopen(line: str) -> str:
    """Match `data_utils.ReadOpen`: commas become spaces, then strip."""
    return " ".join(line.strip().split(","))


def tokenize_tweet(line: str, lowercase: bool = True) -> list[str]:
    cleaned = strip_commas_like_readopen(line)
    tokens = _TOKEN_RE.findall(cleaned)
    if lowercase:
        tokens = [tok.lower() for tok in tokens]
    return tokens


def is_elongated(token: str) -> bool:
    """True when a letter repeats 3+ times (`soooo`) or the token
    looks like the coursework's `loovee` (two separate doubles)."""
    if _ELONG_RE.search(token):
        return True
    doubles = re.findall(r"(.)\1", token.lower())
    return len(doubles) >= 2 and token.isalpha() and len(token) >= 5


def extract_hashtags(tokens: list[str]) -> list[str]:
    return [tok for tok in tokens if tok.startswith("#")]


def extract_emoji(tokens: list[str]) -> list[str]:
    return [tok for tok in tokens if _is_emoji_token(tok)]


def _is_emoji_token(token: str) -> bool:
    if not token:
        return False
    # A token from the emoji branch of _TOKEN_RE is a single glyph.
    code = ord(token[0])
    return (
        0x1F300 <= code <= 0x1FAFF
        or 0x2700 <= code <= 0x27BF
        or 0x2600 <= code <= 0x26FF
    )
