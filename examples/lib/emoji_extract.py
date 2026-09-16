"""Emoji / high-codepoint helpers used by the dataset census and PMI script.

We deliberately do not depend on the `emoji` package. The original Preprocess
walk used emoji.emoji_list + emoji.is_emoji; here a conservative Unicode
range list is enough to study the checked-in dump.
"""

from __future__ import annotations

import re

# Common emoji + dingbat / enclosed-alphanumeric / extra-symbol blocks that
# show up in the 2023 Twitter dump. Keep as a class string so tweet_tokenize
# can splice it into a larger regex.
EMOJI_CHAR_CLASS = (
    r"[\U0001F300-\U0001FAFF"
    r"\U00002700-\U000027BF"
    r"\U00002600-\U000026FF"
    r"\U00002300-\U000023FF"
    r"\U0001F000-\U0001F0FF"
    r"\U0001F100-\U0001F1FF"
    r"\u200d\ufe0f\u20e3]"
)

HIGH_CODEPOINT_FLOOR = 0x2710  # matches the inspect filter described in docs
_VS16 = "\ufe0f"
_GLUE_VS = re.compile(
    r"(["
    r"\U0001F300-\U0001FAFF"
    r"\U00002700-\U000027BF"
    r"\U00002600-\U000026FF"
    r"\U00002300-\U000023FF"
    r"])\s+\ufe0f"
)


def is_emoji_char(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        0x1F300 <= code <= 0x1FAFF
        or 0x2700 <= code <= 0x27BF
        or 0x2600 <= code <= 0x26FF
        or 0x2300 <= code <= 0x23FF
        or 0x1F000 <= code <= 0x1F1FF
        or char in {"\u200d", "\ufe0f", "\u20e3"}
    )


def _is_primary_emoji(char: str) -> bool:
    return is_emoji_char(char) and char not in {"\u200d", _VS16, "\u20e3"}


def extract_emojis(text: str) -> list[str]:
    """Return emoji grapheme clusters (ZWJ sequences kept together).

    Twitter dumps often put a space before U+FE0F (`☺ ️`). Glue that back
    onto the preceding pictograph so PMI tables do not grow a fake `️` row.
    """
    text = _GLUE_VS.sub(lambda m: m.group(1) + _VS16, text)
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        if any(_is_primary_emoji(c) for c in buf):
            out.append("".join(buf))
        buf.clear()

    for char in text:
        if is_emoji_char(char):
            buf.append(char)
        else:
            flush()
    flush()
    return out


def has_high_codepoint(text: str, floor: int = HIGH_CODEPOINT_FLOOR) -> bool:
    return any(ord(char) > floor for char in text)
