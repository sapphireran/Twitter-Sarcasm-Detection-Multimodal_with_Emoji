#!/usr/bin/env python3
"""Print split sizes, class balance, length stats, and cue frequencies."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

# Allow `python3 examples/inspect_dataset.py` without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.features import has_leak_hashtag  # noqa: E402
from lib.io import load_all  # noqa: E402
from lib.tokenize import tokenize_tweet, whitespace_len  # noqa: E402


def _emoji_tokens(tokens):
    return [tok for tok in tokens if len(tok) == 1 and ord(tok) > 255]


def summarize(limit_hashtags: int, limit_emoji: int) -> None:
    print("Split                 n    sarcastic    rate   median_ws   median_tok   leak#   emoji")
    print("-" * 88)
    hashtag_counter = Counter()
    emoji_counter = Counter()
    for split in load_all():
        ws_lens = [whitespace_len(text) for text in split.texts]
        tok_lens = []
        leak = 0
        emoji_tweets = 0
        for text in split.texts:
            tokens = tokenize_tweet(text)
            tok_lens.append(len(tokens))
            if has_leak_hashtag(text):
                leak += 1
            emojis = _emoji_tokens(tokens)
            if emojis:
                emoji_tweets += 1
            if split.name == "train":
                for tok in tokens:
                    if tok.startswith("#"):
                        hashtag_counter[tok] += 1
                emoji_counter.update(emojis)
        print(
            f"{split.name:<12} {len(split):8d}  {sum(split.labels):10d}  "
            f"{split.positive_rate:6.3f}  {statistics.median(ws_lens):9.1f}  "
            f"{statistics.median(tok_lens):10.1f}  {leak:6d}  {emoji_tweets:6d}"
        )

    print("\nTop training hashtags")
    for tag, count in hashtag_counter.most_common(limit_hashtags):
        print(f"  {count:5d}  {tag}")

    print("\nTop training emoji (tokenizer view)")
    for emo, count in emoji_counter.most_common(limit_emoji):
        print(f"  {count:5d}  {emo}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-hashtags", type=int, default=12)
    parser.add_argument("--top-emoji", type=int, default=12)
    args = parser.parse_args()
    summarize(args.top_hashtags, args.top_emoji)


if __name__ == "__main__":
    main()
