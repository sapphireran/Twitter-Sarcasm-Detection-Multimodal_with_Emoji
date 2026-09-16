"""Lightweight tweet tokenizer used by the examples.

The original project tokenizes with ``nltk.TweetTokenizer`` and then
lowercases every token (see ``data_utils.ReadOpen``). This module is a
small, dependency-free stand-in that keeps the tokens that matter for
sarcasm cues: mentions, hashtags, URLs, and emoji.
"""

from __future__ import annotations

import re

# Mentions / hashtags first so "#not" stays one token (a common sarcasm marker
# in this corpus). Emoji next so ZWJ sequences are not split into letters.
# Then URLs, then words (including simple contractions), then leftover punct.
_TOKEN_RE = re.compile(
    r"(?P<url>https?://\S+|www\.\S+)"
    r"|(?P<placeholder></?\w+>)"
    r"|(?P<tag>[@#][\w_]+)"
    r"|(?P<emoji>[\U0001F1E6-\U0001F1FF]{2}"
    r"|[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F"
    r"\U0001F3FB-\U0001F3FF\u200d]+)"
    r"|(?P<word>[A-Za-z0-9]+(?:['’][A-Za-z]+)?)"
    r"|(?P<other>[^\s])"
)


def tokenize_tweet(text: str, lowercase: bool = True) -> list[str]:
    """Split a tweet into tokens, matching the project's lowercase convention.

    Parameters
    ----------
    text:
        Raw tweet text. Callers that want to mimic ``ReadOpen`` should first
        rebuild the line with ``' '.join(line.strip().split(','))``.
    lowercase:
        If True (default), lowercase every token the way ``ReadOpen`` does.
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


def join_comma_split_line(line: str) -> str:
    """Reproduce the sentence rebuild step inside ``data_utils.ReadOpen``.

    The original reader does not use the ``csv`` module. A quoted comma
    therefore becomes a space, and the surrounding quotes stay in the
    string. Examples document this quirk on purpose.
    """
    return " ".join(line.strip().split(","))
