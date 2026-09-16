"""Emoji extraction used by the dataset notes and heuristic baseline.

The original pipeline looks up emoji in ``emoji2vec`` after NLTK tokenization.
These helpers only need the Unicode ranges that appear in the course corpus so
examples can count emoji without the ``emoji`` PyPI package.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# Covers the pictographs that actually show up in this dataset, including
# miscellaneous symbols (☀, ⛳) and the variation selector that follows some
# glyphs. Zero-width joiners stay attached when they appear inside a match.
EMOJI_RE = re.compile(
    r"(?:"
    r"[\U0001F300-\U0001FAFF]"
    r"|[\U00002600-\U000026FF]"
    r"|[\U00002700-\U000027BF]"
    r")[\U0000FE0E\U0000FE0F]?"
    r"(?:\u200D(?:[\U0001F300-\U0001FAFF]|[\U00002600-\U000027BF])[\U0000FE0E\U0000FE0F]?)*"
)


def extract_emojis(text: str) -> list[str]:
    """Return emoji tokens in the order they appear."""
    if not text:
        return []
    return EMOJI_RE.findall(text)


def has_emoji(text: str) -> bool:
    return bool(text and EMOJI_RE.search(text))


def count_emojis(texts: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(extract_emojis(text))
    return counts


# Emoji that often mark irritation, disappointment, or deadpan delivery in this
# corpus. Used only by the educational heuristic, not by the trained models.
NEGATIVE_EMOJI = frozenset(
    {
        "😒",
        "😑",
        "🙄",
        "😩",
        "😭",
        "😡",
        "😠",
        "😣",
        "😕",
        "😞",
        "😔",
        "💔",
        "👎",
        "🔫",
        "💩",
        "😅",
        "😤",
        "😪",
        "😫",
        "😬",
    }
)

POSITIVE_EMOJI = frozenset(
    {
        "😍",
        "😘",
        "😊",
        "😁",
        "😄",
        "😃",
        "❤",
        "💕",
        "💗",
        "✨",
        "😎",
        "😉",
        "🥰",
        "☺",
        "🙂",
    }
)
