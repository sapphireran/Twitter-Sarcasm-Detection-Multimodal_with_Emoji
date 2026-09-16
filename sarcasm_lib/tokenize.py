"""A small Twitter-oriented tokenizer used by the docs and examples.

The original project tokenizes with ``nltk.TweetTokenizer``. That dependency is
not required for the walkthrough scripts. This module keeps mentions, hashtags,
URLs, and emoji as atomic tokens and lowercases by default so the examples stay
close to ``data_utils.ReadOpen``.
"""

from __future__ import annotations

import re
from typing import Iterable

from .emoji import EMOJI_RE

# Order is significant: URLs and Twitter objects must win over word splits.
_TOKEN_RE = re.compile(
    r"(?P<url>https?://\S+|www\.\S+)"
    r"|(?P<placeholder><\w+>)"
    r"|(?P<mention>@\w+)"
    r"|(?P<hashtag>#\w+)"
    r"|(?P<emoji>" + EMOJI_RE.pattern + r")"
    r"|(?P<word>[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?)"
    r"|(?P<punct>[!?.,:;\"'`()\[\]]+)"
    r"|(?P<other>[^\s])"
)

_WHITESPACE_RE = re.compile(r"\s+")


def tokenize_tweet(text: str, *, lowercase: bool = True) -> list[str]:
    """Split a tweet into tokens without NLTK.

    Parameters
    ----------
    text:
        Raw tweet text, possibly still containing ``<user>`` placeholders.
    lowercase:
        Match ``ReadOpen``, which lowercases every NLTK token.
    """
    if not text:
        return []

    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if lowercase:
            # Keep emoji and most symbols intact; only case-fold word-like text.
            if match.lastgroup in {"url", "placeholder", "mention", "hashtag", "word"}:
                token = token.lower()
        tokens.append(token)
    return tokens


def tokenize_corpus(texts: Iterable[str], *, lowercase: bool = True) -> list[list[str]]:
    return [tokenize_tweet(text, lowercase=lowercase) for text in texts]


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def hashtags(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token.startswith("#") and len(token) > 1]


def mentions(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token.startswith("@") or token == "<user>"]
