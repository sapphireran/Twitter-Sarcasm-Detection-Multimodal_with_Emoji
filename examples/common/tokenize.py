"""A small tweet tokenizer used by the examples.

The 2023 notebooks call ``nltk.TweetTokenizer``. That tokenizer is not
available in a numpy-only environment, so this module approximates the same
surface behavior:

* keep hashtags (``#not``) and placeholders (``<user>``) as single tokens
* keep emoji code points as their own tokens
* lowercase after splitting
* treat punctuation as separate tokens except inside hashtags / mentions

This is for documentation and offline experiments. Training still goes through
``data_utils.ReadOpen``.
"""

from __future__ import annotations

import re
from typing import List

from .emoji import EMOJI_RE

# Placeholders used in the CCS2 export of the Twitter data.
PLACEHOLDER = r"<user>|<url>|<URL>"
HASHTAG = r"#\w+"
MENTION = r"@\w+"
# Keep elongated punctuation like "!!!" or "..." together.
PUNCT = r"[!?.]{2,}|[][()\"'`´,;:/\-]+"
WORD = r"[A-Za-z0-9_']+"
TOKEN_RE = re.compile(
    rf"(?:{PLACEHOLDER}|{HASHTAG}|{MENTION}|{WORD}|{PUNCT}|{EMOJI_RE.pattern})",
    re.IGNORECASE | re.UNICODE,
)

HASHTAG_TOKEN_RE = re.compile(r"^#\w+$", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
COMMA_SPLIT_RE = re.compile(r",")

# Cue hashtags that leak the sarcasm label in many Twitter sarcasm corpora.
CUE_HASHTAGS = frozenset(
    {
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#not",
        "#notreally",
        "#yeahright",
        "#shocker",
    }
)


def tokenize_tweet(text: str, *, lowercase: bool = True) -> List[str]:
    """Tokenize one tweet.

    Mirrors ``data_utils.ReadOpen`` by first turning commas into spaces, then
    splitting with the regex above.
    """
    cleaned = COMMA_SPLIT_RE.sub(" ", text.strip())
    tokens = TOKEN_RE.findall(cleaned)
    if lowercase:
        tokens = [tok.lower() for tok in tokens]
    return tokens


def strip_hashtags(text: str) -> str:
    """Drop ``#word`` tokens. Useful for cue-leakage experiments."""
    return re.sub(r"#\w+", " ", text)


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def has_cue_hashtag(tokens: List[str]) -> bool:
    return any(tok in CUE_HASHTAGS for tok in tokens)
