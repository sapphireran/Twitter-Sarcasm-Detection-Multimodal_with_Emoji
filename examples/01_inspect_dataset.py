#!/usr/bin/env python3
"""Inspect the local train/test/subtest CSVs.

Prints split sizes, token/char length stats, emoji and hashtag rates by
label, and the most common hashtags. This is the first example to run:
it only needs the files already in ``dataset/``.
"""

from __future__ import annotations

from collections import Counter

import _path  # noqa: F401

from sarcasm_toolkit.cues import CUE_HASHTAGS
from sarcasm_toolkit.dataset import load_split, summarize_split
from sarcasm_toolkit.tokenize import emoji_tokens, hashtags, tokenize_tweet


def rate(count: int, n: int) -> str:
    return f"{count}/{n} ({count / n:.3f})" if n else "0"


def inspect(name: str, sample: int = 2) -> None:
    split = load_split(name)
    summary = summarize_split(split)
    print(f"\n=== {name} ===")
    print(
        f"n={summary['n']}  sarcastic={summary['sarcastic']} "
        f"({summary['sarcastic_rate']:.3f})  literal={summary['literal']}"
    )
    print(
        "tokens: min={min} median={median} mean={mean:.1f} max={max}".format(
            **summary["token_len"]
        )
    )
    print(
        "chars:  min={min} median={median} mean={mean:.1f} max={max}".format(
            **summary["char_len"]
        )
    )

    by_label = {0: [], 1: []}
    hash_counter: Counter[str] = Counter()
    emoji_counter: Counter[str] = Counter()
    cue_hits = {0: 0, 1: 0}
    emoji_hits = {0: 0, 1: 0}
    hash_hits = {0: 0, 1: 0}
    user_hits = 0

    for example in split:
        tokens = tokenize_tweet(example.text)
        tags = hashtags(tokens)
        emojis = emoji_tokens(tokens)
        hash_counter.update(tags)
        emoji_counter.update(emojis)
        if tags:
            hash_hits[example.label] += 1
        if emojis:
            emoji_hits[example.label] += 1
        if set(tags) & CUE_HASHTAGS:
            cue_hits[example.label] += 1
        if "<user>" in tokens or any(tok.startswith("@") for tok in tokens):
            user_hits += 1
        if len(by_label[example.label]) < sample:
            by_label[example.label].append(example.text)

    n = len(split)
    print(f"<user>/@ mentions: {rate(user_hits, n)}")
    for label, title in ((0, "literal"), (1, "sarcastic")):
        subset_n = summary["literal"] if label == 0 else summary["sarcastic"]
        print(
            f"{title:10s}  emoji={rate(emoji_hits[label], subset_n)}  "
            f"hashtag={rate(hash_hits[label], subset_n)}  "
            f"cue_hashtag={rate(cue_hits[label], subset_n)}"
        )
    print("top hashtags:", hash_counter.most_common(12))
    print("top emoji:   ", emoji_counter.most_common(8))
    print("sample literal:  ", by_label[0][0][:140] if by_label[0] else "")
    print("sample sarcastic:", by_label[1][0][:140] if by_label[1] else "")


def main() -> None:
    print("Dataset inspection (local CSVs only; no network)")
    for name in ("train", "test", "subtest"):
        inspect(name)
    print("\nCue hashtags used by the lexicon example:")
    print(" ", ", ".join(sorted(CUE_HASHTAGS)))


if __name__ == "__main__":
    main()
