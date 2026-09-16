"""Emoji extraction that does not depend on the ``emoji`` PyPI package.

The original preprocessor (``data_utils.Preprocess``) uses ``emoji.emoji_list``
to pull pictographs out of a token and look them up in emoji2vec. The examples
use a Unicode-range heuristic that covers the pictographs actually present in
this dataset (Smileys, People, Food, Travel, Symbols, variation selectors).
"""

from __future__ import annotations

import re
from typing import List

# Broad but practical coverage of emoji code points seen in this corpus.
# Includes:
#   - Miscellaneous Symbols and Dingbats (U+2600–U+27BF)
#   - Enclosed characters used as emoji (U+24C2, U+1F170–)
#   - Emoticons / pictographs (U+1F300–U+1FAFF)
#   - Variation selector-16 (U+FE0F) so we can count "❤︎" style sequences
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # pictographs, faces, animals, food, ...
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flag letters, e.g. 🇺🇸)
    "\U0001F000-\U0001F02F"
    "\U0001F170-\U0001F251"  # enclosed characters used as emoji
    "\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U0000231A-\U0000231B"
    "\U000023E9-\U000023F3"
    "\U000023F8-\U000023FA"
    "\U00002B50"
    "\U00002B55"
    "\U000000A9"
    "\U000000AE"
    "]+",
    flags=re.UNICODE,
)


def is_emoji(char: str) -> bool:
    if not char:
        return False
    return EMOJI_RE.fullmatch(char) is not None or (
        len(char) == 1 and EMOJI_RE.search(char) is not None
    )


def extract_emojis(text: str) -> List[str]:
    """Return pictograph characters in order of appearance.

    Variation selectors and ZWJ are kept attached when they appear next to a
    pictograph so ``❤`` + ``️`` can be inspected together. Callers that want
    single code points can iterate the returned strings.
    """
    return EMOJI_RE.findall(text)


def emoji_code_points(text: str) -> List[str]:
    """Flatten extracted sequences into individual non-selector characters."""
    out: List[str] = []
    for seq in extract_emojis(text):
        for char in seq:
            code = ord(char)
            if code in (0xFE0F, 0xFE0E, 0x200D):
                continue
            out.append(char)
    return out
