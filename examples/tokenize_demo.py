#!/usr/bin/env python3
"""Show how a tweet is rebuilt and tokenized, including the ReadOpen quirk.

Run from the repository root:

    python examples/tokenize_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dataset_io import read_sentence_strings
from examples.emoji import extract_emojis, tokens_that_are_emoji
from examples.tokenize import join_comma_split_line, tokenize_tweet

SAMPLES = [
    "I loovee when people text back ... 😒 #sarcastictweet",
    "Don't you love it when your parents are Pissed ! #IKnowIDo #not 😃 🔫",
    '"So many useless classes , great to be student"',
    "<user> Rest in peace & love to you and your family",
]


def describe(raw: str) -> None:
    rebuilt = join_comma_split_line(raw)
    tokens = tokenize_tweet(rebuilt)
    print(f"raw:      {raw}")
    print(f"rebuilt:  {rebuilt}")
    print(f"tokens:   {tokens}")
    print(f"emoji:    {extract_emojis(raw)}")
    print(f"e-tokens: {tokens_that_are_emoji(tokens)}")
    print()


def main() -> int:
    print("Hard-coded samples")
    print("==================")
    for sample in SAMPLES:
        describe(sample)

    fixture = Path(__file__).resolve().parent / "fixtures" / "tiny_tweets.csv"
    print(f"First three rows of {fixture.name}")
    print("==============================")
    for text in read_sentence_strings(fixture)[:3]:
        describe(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
