"""A small TweetTokenizer stand-in plus the ReadOpen comma-join step.

The 2023 pipeline called ``nltk.TweetTokenizer`` then lowercased every
token. This module keeps the same *shape* of output (a list of lowercase
strings, emoji and hashtags kept whole) so examples can run without NLTK.
It is not a byte-for-byte clone of NLTK's regexes.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from .dataset_io import is_emoji_char

# URLs, mentions, hashtags, words (incl. apostrophes / elongations), leftover punct.
_TOKEN_RE = re.compile(
    r"(https?://\S+)"
    r"|(?<!\w)@\w+"
    r"|(?<!\w)#\w+"
    r"|<user>"
    r"|[A-Za-z0-9]+(?:'[A-Za-z]+)?(?:[A-Za-z0-9]+)*"
    r"|[^\s]"
)


def join_commas(line: str) -> str:
    """Reproduce ``' '.join(line.strip().split(','))`` from ``ReadOpen``."""
    return " ".join(line.strip().split(","))


def tokenize_tweet(text: str, lowercase: bool = True) -> List[str]:
    joined = join_commas(text)
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(joined):
        raw = match.group(0)
        if raw.strip() == "":
            continue
        code = ord(raw) if len(raw) == 1 else None
        # Variation selectors attach to the previous emoji token.
        if code is not None and 0xFE00 <= code <= 0xFE0F and tokens:
            tokens[-1] = tokens[-1] + raw
            continue
        if code is not None and is_emoji_char(raw):
            tokens.append(raw)
            continue
        tokens.append(raw.lower() if lowercase else raw)
    return tokens


def tokenize_corpus(sentences: Sequence[str]) -> List[List[str]]:
    return [tokenize_tweet(sentence) for sentence in sentences]


def token_preview(sentence: str) -> str:
    tokens = tokenize_tweet(sentence)
    return f"{sentence}\n  → {tokens}"


def flatten(docs: Iterable[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for doc in docs:
        out.extend(doc)
    return out
