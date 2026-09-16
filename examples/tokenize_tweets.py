#!/usr/bin/env python3
"""Show Tweet-aware tokenization next to naive whitespace splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset_io import load_all_splits
from examples.lib.tweet_tokenize import tokenize_tweet, whitespace_tokens

DEFAULTS = [
    "I loovee when people text back ... 😒 #sarcastictweet",
    "Oh how I love getting home from work at 3am and my house being dirty #not",
    "<user> Rest in peace & love to you and your family",
    "100 days until Christmas! 🌲 #too soon #not ready yet",
    "Don't you love it when your parents are Pissed 😃🔫",
]


def show(text: str) -> None:
    print(f"TEXT: {text}")
    print(f"  ws    : {whitespace_tokens(text)}")
    print(f"  tweet : {tokenize_tweet(text)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--n", type=int, default=0, help="also sample N train tweets")
    parser.add_argument("--from-split", choices=("train", "test", "subtest"))
    args = parser.parse_args(argv)

    print("## Hand-picked illustrations\n")
    for text in DEFAULTS:
        show(text)

    if args.n > 0:
        split = load_all_splits(args.root)[args.from_split or "train"]
        print(f"## First {args.n} rows from {split.name}\n")
        for text in split.texts[: args.n]:
            show(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
