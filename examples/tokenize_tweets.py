#!/usr/bin/env python3
"""Show tokenizer output and the legacy comma-join quirk on real tweets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarcasm_lib.io import load_split, read_sentences, read_sentences_legacy
from sarcasm_lib.paths import SPLIT_FILES
from sarcasm_lib.tokenize import tokenize_tweet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="subtest",
        choices=sorted(SPLIT_FILES),
        help="which official split to sample",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="how many tweets that contain a comma to display",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split = load_split(args.split)
    modern = read_sentences(SPLIT_FILES[args.split]["sentences"])
    legacy = read_sentences_legacy(SPLIT_FILES[args.split]["sentences"])

    mismatches = [
        (modern_text, legacy_text, label)
        for modern_text, legacy_text, label in zip(modern, legacy, split.labels, strict=True)
        if modern_text != legacy_text
    ]
    print(
        f"{args.split}: {len(mismatches)} tweets change when ReadOpen splits on commas "
        f"(out of {len(split)})"
    )
    print()

    shown = 0
    for modern_text, legacy_text, label in mismatches:
        print(f"label={label}")
        print(f"  csv-aware: {modern_text}")
        print(f"  legacy:    {legacy_text}")
        print(f"  tokens:    {tokenize_tweet(modern_text)}")
        print()
        shown += 1
        if shown >= args.limit:
            break

    if shown == 0:
        print("No comma mismatches in this split. Tokenizing the first tweets instead:")
        for text, label in list(split.pairs())[: args.limit]:
            print(f"label={label}  tokens={tokenize_tweet(text)}")


if __name__ == "__main__":
    main()
