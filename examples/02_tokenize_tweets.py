#!/usr/bin/env python3
"""Show how the example tokenizer splits tweets vs a naive whitespace split.

Walks the curated ``sample_tweets.csv`` first, then a few rows from each
dataset split, and reports tokens that matter for sarcasm cues: hashtags,
emoji, elongated spellings, and ``<user>`` placeholders.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _path  # noqa: F401

from sarcasm_toolkit.cues import cue_hashtags_in, extract_cue_features, feature_names
from sarcasm_toolkit.dataset import load_split
from sarcasm_toolkit.tokenize import (
    emoji_tokens,
    has_elongation,
    hashtags,
    mentions,
    tokenize_tweet,
)

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_tweets.csv"


def show(text: str, label: str | None = None, note: str | None = None) -> None:
    tokens = tokenize_tweet(text)
    features = extract_cue_features(text, tokens=tokens)
    named = dict(zip(feature_names(), features.as_list()))
    header = "TWEET" if label is None else f"label={label}"
    print(f"\n[{header}] {text}")
    if note:
        print(f"  note:     {note}")
    print(f"  tokens:   {tokens}")
    print(f"  hashtags: {hashtags(tokens)}")
    print(f"  emoji:    {emoji_tokens(tokens)}")
    print(f"  mentions: {mentions(tokens)}")
    print(f"  cues:     {cue_hashtags_in(text)}")
    print(f"  elongate: {has_elongation(text)}")
    active = {k: v for k, v in named.items() if v and k not in {"token_count", "char_count"}}
    print(f"  flags:    {active}")


def main() -> None:
    print("Tokenizer walkthrough")
    print("=====================")
    print("Whitespace split keeps '#not 😒' glued or split on punctuation.")
    print("This tokenizer keeps hashtags and emoji as their own tokens.")

    with SAMPLE_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            show(row["text"], label=row["label"], note=row.get("note"))

    print("\n--- first sarcastic and literal tweet from each split ---")
    for name in ("train", "test", "subtest"):
        split = load_split(name)
        seen = {0: False, 1: False}
        print(f"\n{name}:")
        for example in split:
            if seen[example.label]:
                continue
            show(example.text, label=str(example.label))
            seen[example.label] = True
            if all(seen.values()):
                break


if __name__ == "__main__":
    main()
