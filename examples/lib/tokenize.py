"""A small Twitter-aware tokenizer.

This is **not** a drop-in replacement for ``nltk.TweetTokenizer``. It exists so
the walkthroughs can run with zero third-party packages while still keeping
hashtags, @mentions, ``<user>``, urls, and emoji as single tokens — the same
units GloVe-Twitter and emoji2vec care about.
"""

from __future__ import annotations

import re
from typing import List

# Order matters: longer / more specific patterns first.
_TOKEN_RE = re.compile(
    r"(?P<user><user>)"
    r"|(?P<mention>@\w+)"
    r"|(?P<hashtag>#\w+)"
    r"|(?P<url>https?://\S+|www\.\S+)"
    r"|(?P<heart><3)"
    r"|(?P<emoji>[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF])"
    r"|(?P<word>[A-Za-z0-9][A-Za-z0-9']*)"
    r"|(?P<punct>[^\s])"
)


def tokenize_tweet(text: str, lowercase: bool = True) -> List[str]:
    """Return a list of tokens. Empty input yields an empty list."""
    if not text:
        return []
    tokens = [match.group(0) for match in _TOKEN_RE.finditer(text)]
    if lowercase:
        tokens = [tok.lower() for tok in tokens]
    return tokens


def whitespace_len(text: str) -> int:
    return len(text.split()) if text.strip() else 0
