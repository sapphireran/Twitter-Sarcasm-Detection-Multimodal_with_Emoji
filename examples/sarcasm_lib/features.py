"""Lexical feature extraction and association scores for sarcasm cues."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .dataset import Split
from .tokenize import extract_emoji, extract_hashtags, tokenize_tweet

SUPERVISION_TAGS = frozenset(
    {
        "#not",
        "#sarcasm",
        "#sarcastic",
        "#sarcastictweet",
        "#yeahright",
        "#yeah_right",
    }
)

_NOT_RE = re.compile(r"#not\b", re.IGNORECASE)
_SARCASM_TAG_RE = re.compile(
    r"#(sarcasm|sarcastic|sarcastictweet|yeahright|yeah_right)\b",
    re.IGNORECASE,
)


def has_supervision_tag(text: str) -> bool:
    return bool(_NOT_RE.search(text) or _SARCASM_TAG_RE.search(text))


def strip_supervision_tags(text: str) -> str:
    text = _NOT_RE.sub(" ", text)
    text = _SARCASM_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class CueFlags:
    has_emoji: bool
    has_hashtag: bool
    has_user: bool
    has_not_hashtag: bool
    has_explicit_sarcasm_tag: bool
    has_url: bool


def cue_flags(text: str) -> CueFlags:
    tags = extract_hashtags(text)
    tag_set = set(tags)
    return CueFlags(
        has_emoji=bool(extract_emoji(text)),
        has_hashtag=bool(tags),
        has_user="<user>" in text.lower() or bool(re.search(r"@\w+", text)),
        has_not_hashtag="#not" in tag_set,
        has_explicit_sarcasm_tag=any(
            tag in SUPERVISION_TAGS and tag != "#not" for tag in tag_set
        ),
        has_url="<url>" in text.lower() or bool(re.search(r"https?://", text)),
    )


def lexical_tokens(
    text: str,
    *,
    strip_tags: bool = False,
    include_cues: bool = True,
) -> list[str]:
    working = strip_supervision_tags(text) if strip_tags else text
    tokens = tokenize_tweet(working)
    if include_cues:
        flags = cue_flags(working if strip_tags else text)
        if flags.has_emoji:
            tokens.append("CUE:emoji")
        if flags.has_user:
            tokens.append("CUE:user")
        if flags.has_url:
            tokens.append("CUE:url")
        if not strip_tags and flags.has_not_hashtag:
            tokens.append("CUE:not_tag")
        if not strip_tags and flags.has_explicit_sarcasm_tag:
            tokens.append("CUE:sarcasm_tag")
    return tokens


@dataclass
class AssociationRow:
    token: str
    count_neg: int
    count_pos: int
    log_odds: float
    z_score: float


def informative_log_odds(
    counts_neg: Counter[str],
    counts_pos: Counter[str],
    *,
    min_count: int = 5,
) -> list[AssociationRow]:
    """Monroe et al. (2008) informative Dirichlet log-odds.

    Positive ``log_odds`` means the token is more associated with class 1.
    """
    vocab = set(counts_neg) | set(counts_pos)
    n_neg = sum(counts_neg.values())
    n_pos = sum(counts_pos.values())
    # Add-one prior over the observed vocabulary.
    alpha_tok = 1.0
    alpha_neg = alpha_tok * len(vocab)
    alpha_pos = alpha_tok * len(vocab)
    rows: list[AssociationRow] = []
    for token in vocab:
        y0 = counts_neg[token]
        y1 = counts_pos[token]
        if y0 + y1 < min_count:
            continue
        lo0 = math.log(y0 + alpha_tok) - math.log(n_neg + alpha_neg - y0 - alpha_tok)
        lo1 = math.log(y1 + alpha_tok) - math.log(n_pos + alpha_pos - y1 - alpha_tok)
        delta = lo1 - lo0
        var = (
            1.0 / (y0 + alpha_tok)
            + 1.0 / (y1 + alpha_tok)
        )
        z_score = delta / math.sqrt(var)
        rows.append(
            AssociationRow(
                token=token,
                count_neg=y0,
                count_pos=y1,
                log_odds=delta,
                z_score=z_score,
            )
        )
    rows.sort(key=lambda row: row.log_odds, reverse=True)
    return rows


def count_tokens(texts: Iterable[str], extractor) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(extractor(text))
    return counts


@dataclass
class SplitCues:
    n: int
    n_pos: int
    n_neg: int
    emoji_pos: int
    emoji_neg: int
    hashtag_pos: int
    hashtag_neg: int
    not_tag_pos: int
    not_tag_neg: int
    user_pos: int
    user_neg: int
    mean_words_pos: float
    mean_words_neg: float
    emoji_counts_pos: Counter[str] = field(default_factory=Counter)
    emoji_counts_neg: Counter[str] = field(default_factory=Counter)
    hashtag_counts_pos: Counter[str] = field(default_factory=Counter)
    hashtag_counts_neg: Counter[str] = field(default_factory=Counter)


def summarize_cues(split: Split) -> SplitCues:
    emoji_pos: Counter[str] = Counter()
    emoji_neg: Counter[str] = Counter()
    hash_pos: Counter[str] = Counter()
    hash_neg: Counter[str] = Counter()
    n_emoji = [0, 0]
    n_hash = [0, 0]
    n_not = [0, 0]
    n_user = [0, 0]
    words = [0, 0]
    n_class = [0, 0]
    for sentence, label in split:
        flags = cue_flags(sentence)
        n_class[label] += 1
        words[label] += len(sentence.split())
        emojis = extract_emoji(sentence)
        tags = extract_hashtags(sentence)
        if emojis:
            n_emoji[label] += 1
        if tags:
            n_hash[label] += 1
        if flags.has_not_hashtag:
            n_not[label] += 1
        if flags.has_user:
            n_user[label] += 1
        if label == 1:
            emoji_pos.update(emojis)
            hash_pos.update(tags)
        else:
            emoji_neg.update(emojis)
            hash_neg.update(tags)
    def mean(total: int, denom: int) -> float:
        return total / denom if denom else 0.0

    return SplitCues(
        n=len(split),
        n_pos=n_class[1],
        n_neg=n_class[0],
        emoji_pos=n_emoji[1],
        emoji_neg=n_emoji[0],
        hashtag_pos=n_hash[1],
        hashtag_neg=n_hash[0],
        not_tag_pos=n_not[1],
        not_tag_neg=n_not[0],
        user_pos=n_user[1],
        user_neg=n_user[0],
        mean_words_pos=mean(words[1], n_class[1]),
        mean_words_neg=mean(words[0], n_class[0]),
        emoji_counts_pos=emoji_pos,
        emoji_counts_neg=emoji_neg,
        hashtag_counts_pos=hash_pos,
        hashtag_counts_neg=hash_neg,
    )


def format_association_table(
    rows: Sequence[AssociationRow],
    *,
    limit: int = 15,
    positive: bool = True,
) -> str:
    chosen = rows if positive else list(reversed(rows))
    lines = [
        "| token | class 0 | class 1 | log-odds | z |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in chosen[:limit]:
        token = row.token.replace("|", "\\|")
        lines.append(
            f"| {token} | {row.count_neg} | {row.count_pos} | "
            f"{row.log_odds:+.3f} | {row.z_score:+.2f} |"
        )
    return "\n".join(lines)
