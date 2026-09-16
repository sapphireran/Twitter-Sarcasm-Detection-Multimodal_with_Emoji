"""Corpus statistics used by ``docs/dataset.md`` and the inspect example."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .emoji import extract_emojis, has_emoji
from .heuristic import SARCASM_HASHTAGS
from .io import DatasetSplit
from .tokenize import hashtags, tokenize_tweet


@dataclass(frozen=True)
class SplitStats:
    name: str
    n: int
    n_sarcastic: int
    n_non_sarcastic: int
    sarcasm_rate: float
    n_with_emoji: int
    n_sarcastic_with_emoji: int
    n_non_sarcastic_with_emoji: int
    n_with_hashtag: int
    n_with_marker_hashtag: int
    marker_precision: float
    top_hashtags: list[tuple[str, int]] = field(default_factory=list)
    top_emoji: list[tuple[str, int]] = field(default_factory=list)
    mean_tokens: float = 0.0
    mean_emoji: float = 0.0


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_split_stats(split: DatasetSplit, *, top_k: int = 12) -> SplitStats:
    hash_counts: Counter[str] = Counter()
    emoji_counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    emoji_lengths: list[int] = []

    n_emoji = n_sarcastic_emoji = n_non_emoji = 0
    n_hashtag = n_marker = marker_true = 0

    for text, label in split.pairs():
        tokens = tokenize_tweet(text)
        token_lengths.append(len(tokens))
        tags = [tag.lower() for tag in hashtags(tokens)]
        emojis = extract_emojis(text)
        hash_counts.update(tags)
        emoji_counts.update(emojis)
        emoji_lengths.append(len(emojis))

        if tags:
            n_hashtag += 1
        if has_emoji(text):
            n_emoji += 1
            if label == 1:
                n_sarcastic_emoji += 1
            else:
                n_non_emoji += 1

        if set(tags) & SARCASM_HASHTAGS:
            n_marker += 1
            if label == 1:
                marker_true += 1

    return SplitStats(
        name=split.name,
        n=len(split),
        n_sarcastic=split.positive_count,
        n_non_sarcastic=split.negative_count,
        sarcasm_rate=_rate(split.positive_count, len(split)),
        n_with_emoji=n_emoji,
        n_sarcastic_with_emoji=n_sarcastic_emoji,
        n_non_sarcastic_with_emoji=n_non_emoji,
        n_with_hashtag=n_hashtag,
        n_with_marker_hashtag=n_marker,
        marker_precision=_rate(marker_true, n_marker),
        top_hashtags=hash_counts.most_common(top_k),
        top_emoji=emoji_counts.most_common(top_k),
        mean_tokens=_rate(sum(token_lengths), len(token_lengths)),
        mean_emoji=_rate(sum(emoji_lengths), len(emoji_lengths)),
    )


def stats_as_dict(stats: SplitStats) -> dict[str, object]:
    return {
        "split": stats.name,
        "n": stats.n,
        "sarcastic": stats.n_sarcastic,
        "non_sarcastic": stats.n_non_sarcastic,
        "sarcasm_rate": round(stats.sarcasm_rate, 4),
        "with_emoji": stats.n_with_emoji,
        "sarcastic_with_emoji": stats.n_sarcastic_with_emoji,
        "with_hashtag": stats.n_with_hashtag,
        "with_marker_hashtag": stats.n_with_marker_hashtag,
        "marker_precision": round(stats.marker_precision, 4),
        "mean_tokens": round(stats.mean_tokens, 2),
        "mean_emoji": round(stats.mean_emoji, 3),
        "top_hashtags": stats.top_hashtags,
        "top_emoji": stats.top_emoji,
    }
