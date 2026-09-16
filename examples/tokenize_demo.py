#!/usr/bin/env python3
"""Show the walkthrough tokenizer on real tweets or on --text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.io import load_split  # noqa: E402
from lib.tokenize import tokenize_tweet  # noqa: E402

DEFAULT_EXAMPLES = [
    "I just love having grungy ass hair 😑 #not",
    "Don't you love it when your parents are Pissed because you were gonna study! #IKnowIDo #not 😃",
    "<user> well that's going to get you what you want. #Not",
    "3 more days and I will be reunited with my best friend! #CantWait 😁 😍",
]


def _print_one(text: str, label=None) -> None:
    tokens = tokenize_tweet(text)
    prefix = f"[{label}] " if label is not None else ""
    print(f"{prefix}{text}")
    print("   tokens:", " | ".join(tokens))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="tokenize this string instead of built-in / dataset examples")
    parser.add_argument("--split", choices=("train", "test", "subtest"), default="subtest")
    parser.add_argument("--limit", type=int, default=0, help="also print this many tweets from --split")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    if args.text:
        _print_one(args.text)
        return

    print("Built-in walkthrough tweets\n")
    for text in DEFAULT_EXAMPLES:
        _print_one(text)

    if args.limit > 0:
        split = load_split(args.split)
        print(f"From dataset/{args.split}_sentence.csv (offset={args.offset}, limit={args.limit})\n")
        end = min(len(split), args.offset + args.limit)
        for i in range(args.offset, end):
            _print_one(split.texts[i], label=split.labels[i])


if __name__ == "__main__":
    main()
