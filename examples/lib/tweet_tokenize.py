"""A small Twitter-aware tokenizer.

The original pipeline uses nltk.TweetTokenizer then lowercases. This module
covers the token classes that matter for the sarcasm dump (mentions, hashtags,
emoji, urls, punctuation runs, ASCII words) so examples can run without NLTK.
"""

from __future__ import annotations

import re

from .emoji_extract import EMOJI_CHAR_CLASS

# Order matters: URLs and handles before bare punctuation.
_TOKEN_RE = re.compile(
    r"(?P<url>https?://[^\s]+|www\.[^\s]+)"
    r"|(?P<user><user>|@\w+)"
    r"|(?P<hashtag>#\w+)"
    r"|(?P<cash>\$[A-Za-z]+)"
    r"|(?P<word>[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)+)"
    r"|(?P<alnum>[A-Za-z]*\d+[A-Za-z0-9%]*)"
    r"|(?P<ascii>[A-Za-z]+)"
    r"|(?P<emoji>" + EMOJI_CHAR_CLASS + r"+)"
    r"|(?P<punct>[!?.,;:\"'`“”‘’…—–-]+)"
    r"|(?P<other>\S)",
    re.UNICODE,
)


def tokenize_tweet(text: str, lowercase: bool = True) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if lowercase:
            # Emoji and most symbols are unchanged by casefolding.
            token = token.casefold()
        tokens.append(token)
    return tokens


def whitespace_tokens(text: str) -> list[str]:
    return text.split()
