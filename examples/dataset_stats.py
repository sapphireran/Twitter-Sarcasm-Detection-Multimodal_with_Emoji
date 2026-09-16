"""Summaries of the real ``dataset/*.csv`` splits. Stdlib + the local tokenizer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .tokenize import (
    EMOJI_RE,
    HASHTAG_RE,
    read_pairs,
    tweet_tokenize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "dataset"

SPLIT_FILES = {
    "train": (DATASET / "train_sentence.csv", DATASET / "train_label.csv"),
    "test": (DATASET / "test_sentence.csv", DATASET / "test_label.csv"),
    "subtest": (DATASET / "subtest_sentence.csv", DATASET / "subtest_label.csv"),
}


@dataclass
class SplitSummary:
    name: str
    n: int
    n_sarcastic: int
    n_literal: int
    sarcasm_rate: float
    emoji_rate: float
    hashtag_rate: float
    sarcasm_marker_rate: float
    mean_tokens: float
    p50_tokens: float
    p90_tokens: float
    top_hashtags: list[tuple[str, int]]


_MARKERS = {
    "#not",
    "#sarcastictweet",
    "#sarcasm",
    "#yeahright",
    "#yayright",
}


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def summarize_pairs(name: str, pairs: Iterable[tuple[str, int]], top_k: int = 8) -> SplitSummary:
    rows = list(pairs)
    n = len(rows)
    n_sarc = sum(label for _, label in rows)
    emoji = 0
    hashtag = 0
    markers = 0
    lengths: list[int] = []
    tag_counts: Counter[str] = Counter()
    for text, _label in rows:
        tokens = tweet_tokenize(text)
        lengths.append(len(tokens))
        if EMOJI_RE.search(text):
            emoji += 1
        tags = [tok for tok in tokens if HASHTAG_RE.fullmatch(tok)]
        if tags:
            hashtag += 1
        tag_counts.update(tags)
        if any(tag in _MARKERS for tag in tags):
            markers += 1
    return SplitSummary(
        name=name,
        n=n,
        n_sarcastic=n_sarc,
        n_literal=n - n_sarc,
        sarcasm_rate=n_sarc / n if n else 0.0,
        emoji_rate=emoji / n if n else 0.0,
        hashtag_rate=hashtag / n if n else 0.0,
        sarcasm_marker_rate=markers / n if n else 0.0,
        mean_tokens=float(sum(lengths) / n) if n else 0.0,
        p50_tokens=_percentile(lengths, 0.50),
        p90_tokens=_percentile(lengths, 0.90),
        top_hashtags=tag_counts.most_common(top_k),
    )


def summarize_split(name: str, top_k: int = 8) -> SplitSummary:
    if name not in SPLIT_FILES:
        raise KeyError(f"unknown split {name!r}; expected one of {sorted(SPLIT_FILES)}")
    sent, lab = SPLIT_FILES[name]
    return summarize_pairs(name, read_pairs(sent, lab), top_k=top_k)


def summarize_all(top_k: int = 8) -> list[SplitSummary]:
    return [summarize_split(name, top_k=top_k) for name in ("train", "test", "subtest")]


def as_plain_dict(summary: SplitSummary) -> dict:
    return asdict(summary)
