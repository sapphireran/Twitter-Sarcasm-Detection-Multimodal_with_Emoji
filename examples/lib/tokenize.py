"""Replay `data_utils.ReadOpen` without requiring NLTK.

`ReadOpen` does two things before `TweetTokenizer`:

1. `sentence = ' '.join(line.strip().split(','))`  — undo CSV commas
2. lowercase each token after NLTK tokenization

NLTK `TweetTokenizer` keeps hashtags, @mentions, emoji, and a few
emoticons as single tokens. The regex below is an approximation used
only by the walkthroughs. It is not claimed to match NLTK byte-for-byte.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .dataset import Split

# Order matters: hashtags / mentions / urls / html user token / emoji
# / words / leftover punctuation.
_TOKEN_RE = re.compile(
    r"(#\w+"
    r"|@\w+"
    r"|<user>"
    r"|https?://\S+"
    r"|[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    r"\U00002600-\U000026FF\U0001F600-\U0001F64F]"
    r"|[A-Za-z0-9']+"
    r"|[^\s])",
    re.UNICODE,
)


def comma_unwrap(line: str) -> str:
    """Same transform as `ReadOpen` before TweetTokenizer."""
    return " ".join(line.strip().split(","))


def tokenize_tweet(line: str) -> List[str]:
    unwrapped = comma_unwrap(line)
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(unwrapped)]


def tokenize_split(split: Split) -> List[List[str]]:
    return [tokenize_tweet(sentence) for sentence in split.sentences]


def pad_sequences(sequences: Iterable[List[int]], maxlen: int) -> List[List[int]]:
    """Right-pad / truncate the way Keras `padding='post'` does."""
    padded: List[List[int]] = []
    for seq in sequences:
        if len(seq) >= maxlen:
            padded.append(seq[:maxlen])
        else:
            padded.append(seq + [0] * (maxlen - len(seq)))
    return padded
