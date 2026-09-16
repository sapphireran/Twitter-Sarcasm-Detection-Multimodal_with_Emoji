"""Tweet-ish tokenizer that follows ``data_utils.ReadOpen`` without NLTK.

Rules copied from the 2023 helper:

1. Strip the line.
2. Split on commas and re-join with spaces (CSV leftovers).
3. Lowercase.
4. Keep hashtags, @mentions, ``<user>``, and urls as single tokens.
5. Keep each emoji / dingbat code point as its own token.
6. Keep word tokens (letters, digits, apostrophes).
7. Keep leftover punctuation as one-character tokens.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# Hashtags, mentions, <user>, urls, words, or a single other character.
_TOKEN_RE = re.compile(
    r"(#\w+"
    r"|@\w+"
    r"|<user>"
    r"|https?://\S+"
    r"|[a-z0-9]+(?:'[a-z0-9]+)*"
    r"|[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]"
    r"|[^\s])",
    re.IGNORECASE,
)

# Broader "does this tweet contain an emoji?" test used by dataset_stats.
EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]"
)
HASHTAG_RE = re.compile(r"#\w+", re.IGNORECASE)


def normalize_line(line: str) -> str:
    """Reproduce the comma-join step in ``ReadOpen``."""
    return " ".join(line.strip().split(","))


def tweet_tokenize(text: str) -> list[str]:
    cleaned = normalize_line(text).lower()
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(cleaned)]


def is_emoji_token(token: str) -> bool:
    return bool(token) and EMOJI_RE.fullmatch(token) is not None


def read_sentence_file(path: str | Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            rows.append(tweet_tokenize(line))
    return rows


def read_label_file(path: str | Path) -> list[int]:
    labels: list[int] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "":
                continue
            labels.append(int(stripped))
    return labels


def read_pairs(
    sentence_path: str | Path, label_path: str | Path
) -> list[tuple[str, int]]:
    with open(sentence_path, encoding="utf-8", errors="replace") as handle:
        sentences = [line.rstrip("\n") for line in handle if line.strip() != ""]
    labels = read_label_file(label_path)
    if len(sentences) != len(labels):
        raise ValueError(
            f"length mismatch: {sentence_path} has {len(sentences)} rows, "
            f"{label_path} has {len(labels)}"
        )
    return list(zip(sentences, labels))


def iter_hashtags(tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if HASHTAG_RE.fullmatch(token)]
