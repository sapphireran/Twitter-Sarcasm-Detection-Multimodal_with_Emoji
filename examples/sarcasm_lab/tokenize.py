"""Tweet-ish tokenizer that does not depend on NLTK.

The original project uses ``nltk.TweetTokenizer``. This splitter keeps the
pieces that matter for the examples: ``<user>``, hashtags, emoji, urls, and
ordinary words. It is not a byte-for-byte NLTK clone.
"""

from __future__ import annotations

import re
import unicodedata

# Mentions were already folded to <user> in the committed files.
USER_TOKEN = "<user>"

_URL = r"https?://\S+|www\.\S+"
_USER = r"<user>"
_HASHTAG = r"#[\w']+"
_MENTION = r"@\w+"
_WORD = r"[\w']+"
_ELLIPSIS = r"\.{2,}|…+"
_OTHER = r"[^\s]"

_TOKEN_RE = re.compile(
    "|".join((_URL, _USER, _HASHTAG, _MENTION, _ELLIPSIS, _WORD, _OTHER)),
    flags=re.UNICODE | re.IGNORECASE,
)

# Common emoji / dingbat / pictograph blocks. Good enough for this dataset
# without the ``emoji`` package.
_EMOJI_RANGES = (
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0x1F1E6, 0x1F1FF),  # regional indicators (flags)
    (0xFE0F, 0xFE0F),  # variation selector-16
    (0x200D, 0x200D),  # ZWJ
)


def is_emoji_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    for start, end in _EMOJI_RANGES:
        if start <= code <= end:
            return True
    return unicodedata.category(char) == "So" and code > 0x2000


def is_emoji_token(token: str) -> bool:
    return bool(token) and all(is_emoji_char(ch) for ch in token)


def is_hashtag(token: str) -> bool:
    return token.startswith("#") and len(token) > 1


def normalize_token(token: str) -> str:
    lowered = token.lower()
    if lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("www."):
        return "<url>"
    return lowered


def tokenize_tweet(text: str) -> list[str]:
    """Lowercase tokens, collapse URLs, keep hashtags and emoji."""
    if not text:
        return []
    # data_utils.ReadOpen joins on commas before tokenizing.
    text = text.replace(",", " ")
    pieces: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        if raw.isspace():
            continue
        # Split runs like "😒#not" if the regex glued them (it should not,
        # but emoji + hashtag without a space shows up in the wild).
        if len(raw) > 1 and is_emoji_char(raw[0]) and not is_emoji_token(raw):
            buf = ""
            for char in raw:
                if is_emoji_char(char):
                    if buf:
                        pieces.append(normalize_token(buf))
                        buf = ""
                    pieces.append(char)
                else:
                    buf += char
            if buf:
                pieces.append(normalize_token(buf))
            continue
        pieces.append(normalize_token(raw))
    return [piece for piece in pieces if piece and piece != "\ufe0f"]
