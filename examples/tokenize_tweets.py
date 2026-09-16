#!/usr/bin/env python3
"""Show tweet tokenization, emoji extraction, and the ReadOpen comma quirk.

The 2023 ``ReadOpen`` function did not use a CSV parser. It stripped each
line, split on commas, and joined the pieces with spaces before calling
``nltk.TweetTokenizer``. This script places that transform next to a
quote-aware reading of the same line so the preprocessing write-up has a
concrete before/after.

Usage:

    python3 examples/tokenize_tweets.py
    python3 examples/tokenize_tweets.py --n 6 --split train
    python3 examples/tokenize_tweets.py --text "I love Monday mornings #not 😒"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset import load_split
from examples.lib.tokenize import extract_emojis, extract_hashtags, tokenize_tweet


def _show(title: str, text: str) -> None:
    replay = " ".join(text.strip().split(","))
    print(f"--- {title}")
    print(f"raw:            {text}")
    print(f"ReadOpen-style: {replay}")
    print(f"tokens:         {tokenize_tweet(text)}")
    print(f"hashtags:       {extract_hashtags(text)}")
    print(f"emoji:          {extract_emojis(text)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="subtest", choices=("train", "test", "subtest"))
    parser.add_argument("--n", type=int, default=5, help="How many dump rows to show.")
    parser.add_argument("--text", action="append", help="Extra hand-written tweet(s).")
    parser.add_argument(
        "--prefer-emoji",
        action="store_true",
        help="Prefer rows that already contain emoji (useful on the full train split).",
    )
    args = parser.parse_args(argv)

    if args.text:
        for i, text in enumerate(args.text, start=1):
            _show(f"cli-{i}", text)

    split = load_split(args.split)
    shown = 0
    for text in split.texts:
        if args.prefer_emoji and not extract_emojis(text):
            continue
        _show(f"{args.split}-{shown + 1}", text)
        shown += 1
        if shown >= args.n:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
