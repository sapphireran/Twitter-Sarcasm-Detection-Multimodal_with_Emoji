#!/usr/bin/env python3
"""Measure emoji and sarcasm-hashtag association with the label.

The project claims the emoji channel matters most on subtest. This
script checks the raw co-occurrence first, before any embedding
model: P(label=1 | has emoji-like token), lift over the base rate,
and a simple mutual-information style count table.

    python3 examples/emoji_signal.py
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset import load_all  # noqa: E402
from examples.lib.tokenize import tokenize_tweet  # noqa: E402

SARC_TAGS = {"#not", "#sarcasm", "#sarcastic", "#sarcastictweet"}

# Broad "looks like emoji / dingbat" test used when we do not have
# the `emoji` package. Matches the non-ASCII diagnostic in docs/.
_EMOJI_RANGES = (
    (0x2600, 0x27BF),
    (0x1F300, 0x1FAFF),
    (0x1F900, 0x1F9FF),
    (0x1F600, 0x1F64F),
    (0xFE00, 0xFE0F),  # variation selectors
    (0x1F3FB, 0x1F3FF),  # skin tones
)


def token_is_emoji(token: str) -> bool:
    if token.startswith("#") or token.startswith("@") or token == "<user>":
        return False
    for char in token:
        code = ord(char)
        if any(start <= code <= end for start, end in _EMOJI_RANGES):
            return True
        if code > 127 and not char.isalnum():
            # leftover symbols the walkthrough tokenizer kept (™, …, etc.)
            return True
    return False


def tweet_flags(sentence: str) -> Tuple[bool, bool]:
    tokens = tokenize_tweet(sentence)
    has_emoji = any(token_is_emoji(tok) for tok in tokens)
    has_tag = bool({tok for tok in tokens if tok.startswith("#")} & SARC_TAGS)
    return has_emoji, has_tag


def rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def phi(table: Sequence[int]) -> float:
    """Phi coefficient for a 2x2 [n11, n10, n01, n00] count table."""
    n11, n10, n01, n00 = table
    n = n11 + n10 + n01 + n00
    if n == 0:
        return float("nan")
    n1_ = n11 + n10
    n0_ = n01 + n00
    n_1 = n11 + n01
    n_0 = n10 + n00
    denom = math.sqrt(n1_ * n0_ * n_1 * n_0)
    if denom == 0:
        return float("nan")
    return (n11 * n00 - n10 * n01) / denom


def analyze_split(sentences: List[str], labels: List[int]) -> Dict[str, object]:
    n = len(labels)
    pos = sum(labels)
    emoji_pos = emoji_neg = tag_pos = tag_neg = 0
    both_pos = both_neg = 0
    emoji_only_pos = emoji_only_neg = 0
    token_counter: Counter[str] = Counter()
    token_pos: Counter[str] = Counter()

    for sentence, label in zip(sentences, labels):
        has_emoji, has_tag = tweet_flags(sentence)
        tokens = tokenize_tweet(sentence)
        emojis = [tok for tok in tokens if token_is_emoji(tok)]
        for tok in set(emojis):
            token_counter[tok] += 1
            if label == 1:
                token_pos[tok] += 1
        if has_emoji:
            emoji_pos += label
            emoji_neg += 1 - label
        if has_tag:
            tag_pos += label
            tag_neg += 1 - label
        if has_emoji and has_tag:
            both_pos += label
            both_neg += 1 - label
        if has_emoji and not has_tag:
            emoji_only_pos += label
            emoji_only_neg += 1 - label

    n_emoji = emoji_pos + emoji_neg
    n_no_emoji = n - n_emoji
    no_emoji_pos = pos - emoji_pos
    top = []
    for tok, count in token_counter.most_common(12):
        top.append(
            {
                "token": tok,
                "tweets": count,
                "p_sarcastic": rate(token_pos[tok], count),
            }
        )

    return {
        "n": n,
        "base_rate": rate(pos, n),
        "p_sarc_given_emoji": rate(emoji_pos, n_emoji),
        "p_sarc_given_no_emoji": rate(no_emoji_pos, n_no_emoji),
        "emoji_coverage": rate(n_emoji, n),
        "phi_emoji": phi([emoji_pos, emoji_neg, no_emoji_pos, n_no_emoji - no_emoji_pos]),
        "p_sarc_given_tag": rate(tag_pos, tag_pos + tag_neg),
        "p_sarc_given_emoji_only": rate(emoji_only_pos, emoji_only_pos + emoji_only_neg),
        "p_sarc_given_emoji_and_tag": rate(both_pos, both_pos + both_neg),
        "n_emoji": n_emoji,
        "n_tag": tag_pos + tag_neg,
        "n_emoji_only": emoji_only_pos + emoji_only_neg,
        "top_emoji": top,
    }


def _print_split(name: str, block: Dict[str, object]) -> None:
    print(f"=== {name} ===")
    print(f"n                         {block['n']}")
    print(f"base P(sarc)              {block['base_rate']:.3f}")
    print(f"emoji coverage            {block['emoji_coverage']:.3f}  (n={block['n_emoji']})")
    print(f"P(sarc | has emoji)       {block['p_sarc_given_emoji']:.3f}")
    print(f"P(sarc | no emoji)        {block['p_sarc_given_no_emoji']:.3f}")
    print(f"phi(emoji, label)         {block['phi_emoji']:.3f}")
    print(f"P(sarc | sarc hashtag)    {block['p_sarc_given_tag']:.3f}  (n={block['n_tag']})")
    print(
        f"P(sarc | emoji, no tag)   {block['p_sarc_given_emoji_only']:.3f}  "
        f"(n={block['n_emoji_only']})"
    )
    print(f"P(sarc | emoji and tag)   {block['p_sarc_given_emoji_and_tag']:.3f}")
    print("top emoji-like tokens (walkthrough tokenizer):")
    for row in block["top_emoji"]:
        print(
            f"  {row['token']:>4}   tweets={row['tweets']:<5}  "
            f"P(sarc|token)={row['p_sarcastic']:.3f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "test", "subtest"),
        help="restrict to one or more splits (default: all)",
    )
    args = parser.parse_args()
    splits = load_all()
    names = args.split or list(splits)
    print(
        "Emoji / hashtag association with the sarcastic label.\n"
        "Tokenizer is the walkthrough regex, not NLTK TweetTokenizer.\n"
    )
    for name in names:
        split = splits[name]
        _print_split(name, analyze_split(split.sentences, split.labels))
    print(
        "Interpretation: subtest coverage is 1.0 by construction "
        "(every line is non-ASCII). The useful comparison is "
        "P(sarc | emoji, no tag) versus the base rate — that is the "
        "slice where emoji2vec has to do work the hashtag cannot."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
