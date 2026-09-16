"""Hand-built 8-d tables for the toy pipeline.

Axes are named so the demos stay readable:

0  positive surface words (love, happy, great)
1  negative surface words (hate, annoyed, disappointing)
2  sarcasm hashtags (#not, #sarcastictweet, #yeahright)
3  school / work grind words (test, class, monday)
4  genuine affection / family words
5  weeping-face / tired emoji
6  unamused / upside-down affect emoji
7  happy-face emoji that often co-occur with sarcastic praise
"""

from __future__ import annotations

import numpy as np

DIM = 8


def _vec(*pairs: tuple[int, float]) -> np.ndarray:
    out = np.zeros(DIM, dtype=np.float64)
    for index, value in pairs:
        out[index] = value
    return out


# Values are (axis, magnitude) pairs.
WORD_TABLE: dict[str, np.ndarray] = {
    "love": _vec((0, 1.0)),
    "loovee": _vec((0, 1.0)),
    "happy": _vec((0, 0.8)),
    "great": _vec((0, 0.7)),
    "best": _vec((0, 0.6)),
    "hate": _vec((1, 1.0)),
    "annoyed": _vec((1, 0.9)),
    "disappointing": _vec((1, 0.8)),
    "worst": _vec((1, 0.7)),
    "pissed": _vec((1, 0.7)),
    "grungy": _vec((1, 0.4)),
    "#not": _vec((2, 1.0)),
    "#sarcastictweet": _vec((2, 1.0)),
    "#yeahright": _vec((2, 0.9)),
    "#iknowido": _vec((2, 0.7)),
    "test": _vec((3, 0.8)),
    "class": _vec((3, 0.6)),
    "monday": _vec((3, 0.5)),
    "school": _vec((3, 0.5)),
    "chem": _vec((3, 0.4)),
    "peace": _vec((4, 0.9)),
    "family": _vec((4, 0.8)),
    "christmas": _vec((4, 0.5)),
    "dancing": _vec((4, 0.4)),
}

EMOJI_TABLE: dict[str, np.ndarray] = {
    "😭": _vec((5, 1.0)),
    "😅": _vec((5, 0.6)),
    "😔": _vec((5, 0.7)),
    "😑": _vec((6, 0.8)),
    "😒": _vec((6, 1.0)),
    "😄": _vec((7, 0.8)),
    "😃": _vec((7, 0.7)),
    "👏": _vec((7, 0.4)),
    "🌲": _vec((4, 0.3)),
}


def tables() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return WORD_TABLE, EMOJI_TABLE
