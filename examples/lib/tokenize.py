"""A small tweet tokenizer that does not depend on NLTK.

It keeps hashtags, @mentions, <user> placeholders, URLs, and emoji as
single tokens, then lowercases. That is close in spirit to the
``nltk.TweetTokenizer`` pass in ``data_utils.ReadOpen``.
"""

from __future__ import annotations

import re

# Broad but dependency-free emoji coverage: supplemental pictographs,
# dingbats/misc symbols, and regional-indicator flags.
_EMOJI = (
    r"[\U0001F300-\U0001FAFF]"
    r"|[\u2600-\u27BF]"
    r"|[\U0001F1E6-\U0001F1FF]"
)

_TOKEN = re.compile(
    r"(?:"
    r"https?://\S+|www\.\S+"
    r"|@[A-Za-z0-9_]+"
    r"|#[A-Za-z0-9_]+"
    r"|<user>"
    r"|" + _EMOJI + r""
    r"|[A-Za-z0-9]+(?:'[A-Za-z]+)?"
    r"|[^\s]"
    r")",
    re.IGNORECASE,
)

_EMOJI_ONLY = re.compile(_EMOJI)
_HASHTAG = re.compile(r"#[A-Za-z0-9_]+", re.IGNORECASE)

# Explicit sarcasm tags used in the 2023 write-up and in example 02.
SARCASM_TAGS = (
    "#not",
    "#sarcasm",
    "#sarcastictweet",
    "#yeahright",
    "#sarcastic",
)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def find_hashtags(text: str) -> list[str]:
    return [match.group(0).lower() for match in _HASHTAG.finditer(text)]


def find_emoji(text: str) -> list[str]:
    return _EMOJI_ONLY.findall(text)


def has_sarcasm_tag(text: str, tags: tuple[str, ...] = SARCASM_TAGS) -> bool:
    present = set(find_hashtags(text))
    return any(tag in present for tag in tags)
