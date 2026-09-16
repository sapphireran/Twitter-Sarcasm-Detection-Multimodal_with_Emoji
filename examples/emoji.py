"""Emoji detection helpers used by the dataset examples.

``data_utils.Preprocess`` falls back to ``emoji.emoji_list`` when a token
is missing from GloVe and then looks the extracted glyphs up in emoji2vec.
The examples keep a smaller Unicode-range detector so they run without the
``emoji`` package.
"""

from __future__ import annotations

import re

# Covers the blocks that actually appear in this corpus: emoticons, dingbats,
# transport, supplemental symbols, flags, and variation selectors. It is not
# a complete Unicode emoji list.
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE0F"
    "]+",
    re.UNICODE,
)

# Isolated pictographs (used when we want one glyph per match, not runs).
_SINGLE_EMOJI_RE = re.compile(
    r"[\U0001F1E6-\U0001F1FF]{2}"
    r"|[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"
    r"[\U0001F3FB-\U0001F3FF]?"
    r"\U0000FE0F?",
    re.UNICODE,
)


def has_emoji(text: str) -> bool:
    """Return True if ``text`` contains at least one emoji-range code point."""
    return bool(EMOJI_RE.search(text or ""))


def extract_emojis(text: str) -> list[str]:
    """Return the emoji glyphs in ``text``, in order of appearance.

    This is the example-side analogue of the ``emoji.emoji_list`` loop in
    ``data_utils.Preprocess``: walk the string, keep pictographs, ignore
    ordinary words.
    """
    if not text:
        return []
    return _SINGLE_EMOJI_RE.findall(text)


def tokens_that_are_emoji(tokens: list[str]) -> list[str]:
    """Filter a token list to those that look like emoji.

    ``AverageVectorPerEmoji`` in ``data_utils.py`` does not extract emoji
    first. It walks every token and keeps the ones that hit the emoji2vec
    vocabulary. In practice that vocabulary is almost only emoji, so this
    filter is a close stand-in when we do not load the binary vectors.
    """
    return [tok for tok in tokens if has_emoji(tok)]
