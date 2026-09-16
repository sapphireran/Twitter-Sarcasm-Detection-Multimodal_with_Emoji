"""A 16-tweet corpus that makes the multimodal idea visible.

The lines are original examples written for this docs pass. They mimic the
*style* of ``dataset/subtest_sentence.csv`` (fake-positive verbs, emoji,
``#not`` / ``#sarcastictweet``) but they are not copied from the course dump.

Labels: 1 = sarcastic, 0 = literal.
"""

from __future__ import annotations

from typing import Sequence

# (tweet, label)
_PAIRS: list[tuple[str, int]] = [
    ("i just love getting shots 💉 #sarcastictweet", 1),
    ("i love walking to school 😒 #not", 1),
    ("nothing like a 4 hour monday test 😭 #not", 1),
    ("don't you love it when the wifi dies 😡 #sarcastictweet", 1),
    ("i love mondays 😒 #not", 1),
    ("feeling like a million bucks after that chem test 😅 #not", 1),
    ("great news always comes at the right time 😊 #sarcastictweet", 1),
    ("i love when the bus is 20 minutes late 👌 #not", 1),
    ("rest in peace and love to you and your family", 0),
    ("100 days until christmas 🌲 cannot wait", 0),
    ("thank you for the notes from yesterday's lecture", 0),
    ("i love mondays", 0),
    ("the new album is actually really good", 0),
    ("see you at the study group after class", 0),
    ("pretty please let this concert happen someday 😭", 0),
    ("coffee and a quiet library is all i need", 0),
]

TOY_TWEETS: list[str] = [text for text, _ in _PAIRS]
TOY_LABELS: list[int] = [label for _, label in _PAIRS]


def illustrative_pair() -> tuple[str, str]:
    """Return the minimal pair that only the emoji channel can split."""
    sarcastic = "i love mondays 😒 #not"
    literal = "i love mondays"
    if sarcastic not in TOY_TWEETS or literal not in TOY_TWEETS:
        raise RuntimeError("illustrative pair missing from TOY_TWEETS")
    return sarcastic, literal


def labeled_rows() -> Sequence[tuple[str, int]]:
    return tuple(_PAIRS)
