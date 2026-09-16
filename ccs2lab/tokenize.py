"""Tweet-aware tokenizer used by the archive lab.

The 2023 ``ReadOpen`` helper joined comma-separated CSV fields with
spaces, then ran ``nltk.TweetTokenizer`` and lowercased every token.
This module reconstructs that pipeline without NLTK so the examples
stay dependency-light.

It is not a bit-for-bit clone of NLTK. The contract is:

* keep ``#hashtags``, ``@mentions``, ``<user>``, and URLs together
* keep emoji as their own tokens
* keep multi-digit numbers intact (``100``, not ``1 0 0``)
* keep apostrophes inside words (``i'm``, ``don't``)
* lowercase after the split
"""

from __future__ import annotations

import re
from typing import Iterable

# Order matters: URLs and placeholders before loose punctuation.
_TOKEN_RE = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|<user>|<url>|<hashtag>|<number>"
    r"|[@#][\w_]+"
    r"|[\U0001F300-\U0001FAFF\U0001F900-\U0001F9FF"
    r"\U00002700-\U000027BF\U00002600-\U000026FF"
    r"\U0001F1E6-\U0001F1FF]"
    r"|[A-Za-z]+(?:'[A-Za-z]+)+"
    r"|[A-Za-z]+"
    r"|\d+"
    r"|[!?]{2,}"
    r"|\.{3,}"
    r"|[^\s]"
)

_COMMA_SPLIT = re.compile(r",")


def csv_line_to_text(line: str) -> str:
    """Mirror ``ReadOpen``: strip, then turn CSV commas into spaces."""
    return " ".join(_COMMA_SPLIT.split(line.strip()))


def tokenize_tweet(text: str, *, already_csv_normalized: bool = False) -> list[str]:
    """Lowercased tweet tokens. CSV lines should pass ``already_csv_normalized=False``."""
    if not already_csv_normalized:
        text = csv_line_to_text(text)
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def tokenize_many(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize_tweet(text) for text in texts]


def hashtags(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token.startswith("#") and len(token) > 1]


def mentions(tokens: Iterable[str]) -> list[str]:
    return [
        token
        for token in tokens
        if token.startswith("@") or token == "<user>"
    ]
