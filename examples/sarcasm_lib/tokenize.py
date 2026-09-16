"""A small tweet tokenizer used by the docs examples.

It is not NLTK ``TweetTokenizer``. It keeps the token classes the original
pipeline cares about: words, hashtags, anonymized mentions, URL placeholders,
and emoji.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Regional indicators, enclosed ideographs, and the usual pictograph blocks.
_EMOJI_RANGES = (
    (0x2194, 0x2199),
    (0x21A9, 0x21AA),
    (0x231A, 0x231B),
    (0x23E9, 0x23F3),
    (0x23F8, 0x23FA),
    (0x25AA, 0x25AB),
    (0x25B6, 0x25B6),
    (0x25C0, 0x25C0),
    (0x25FB, 0x25FE),
    (0x2600, 0x27BF),
    (0x2934, 0x2935),
    (0x2B05, 0x2B07),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3297),
    (0x3299, 0x3299),
    (0x1F004, 0x1F004),
    (0x1F0CF, 0x1F0CF),
    (0x1F170, 0x1F251),
    (0x1F300, 0x1FAFF),
)

_FITZPATRICK = set(range(0x1F3FB, 0x1F3FF + 1))
_ZWJ = 0x200D
_VS16 = 0xFE0F
_KEYCAP = 0x20E3

_TOKEN_RE = re.compile(
    r"(?P<user><user>)"
    r"|(?P<url><url>|https?://\S+)"
    r"|(?P<hashtag>#[\w']+)"
    r"|(?P<mention>@\w+)"
    r"|(?P<word>[A-Za-z0-9]+(?:'[A-Za-z]+)?)"
    r"|(?P<punct>[^\s\w])",
    re.UNICODE,
)


def is_emoji_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    if code in _FITZPATRICK or code in {_ZWJ, _VS16, _KEYCAP}:
        return True
    for start, end in _EMOJI_RANGES:
        if start <= code <= end:
            return True
    return unicodedata.category(char) == "So" and code >= 0x2000


def extract_emoji(text: str) -> list[str]:
    """Group ZWJ sequences and Fitzpatrick modifiers into single tokens."""
    chars = list(text)
    tokens: list[str] = []
    i = 0
    while i < len(chars):
        if not is_emoji_char(chars[i]):
            i += 1
            continue
        cluster = [chars[i]]
        i += 1
        while i < len(chars) and is_emoji_char(chars[i]):
            cluster.append(chars[i])
            i += 1
        token = "".join(ch for ch in cluster if ord(ch) != _VS16)
        if token:
            tokens.append(token)
    return tokens


def extract_hashtags(text: str) -> list[str]:
    return [match.group(0).lower() for match in re.finditer(r"#[\w']+", text)]


@dataclass(frozen=True)
class TweetishTokenizer:
    lowercase: bool = True
    keep_punct: bool = False

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        length = len(text)
        while index < length:
            if is_emoji_char(text[index]):
                cluster = [text[index]]
                index += 1
                while index < length and is_emoji_char(text[index]):
                    cluster.append(text[index])
                    index += 1
                token = "".join(ch for ch in cluster if ord(ch) != _VS16)
                if token:
                    tokens.append(token)
                continue
            match = _TOKEN_RE.match(text, index)
            if not match:
                index += 1
                continue
            kind = match.lastgroup
            raw = match.group(0)
            index = match.end()
            if kind == "punct" and not self.keep_punct:
                continue
            tokens.append(raw.lower() if self.lowercase else raw)
        return tokens


_DEFAULT = TweetishTokenizer()


def tokenize_tweet(text: str) -> list[str]:
    return _DEFAULT.tokenize(text)
