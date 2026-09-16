#!/usr/bin/env python3
"""Show how this repo turns a raw tweet line into tokens.

``data_utils.ReadOpen`` uses NLTK TweetTokenizer after replacing commas with
spaces. This script does not import NLTK. It prints:

* the raw line
* the comma-collapsed line (exact ReadOpen step)
* whitespace split after that collapse
* the lightweight tweet tokenizer in examples/common.py

Use it to see why emoji, hashtags, and ``<user>`` must stay intact before
GloVe / emoji2vec lookup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    collapse_commas,
    load_split,
    sarcasm_cue_tags,
    simple_tweet_tokens,
    whitespace_tokens,
)

# Hand-picked lines that show the preprocessing decisions. Indices are into
# the official files and are part of the example, not a random sample.
CURATED = {
    "train": [0, 1, 4, 6],
    "test": [0, 1, 2, 3],
    "subtest": [0, 1, 2],
}

# Extra synthetic tweets so the tokenizer can be shown without the CSVs.
SYNTHETIC = [
    'I loovee when people text back ... 😒 #sarcastictweet',
    'Don\'t you love it when your parents are Pissed ! #IKnowIDo #not 😃 🔫',
    '"Late nights, early mornings is how I live my life."',
    "<user> Rest in peace & love to you and your family",
    "100 days until Christmas! 🌲 #too soon #not ready yet",
]


def show_one(title: str, text: str) -> None:
    collapsed = collapse_commas(text)
    white = whitespace_tokens(text)
    tweet = simple_tweet_tokens(text)
    cues = sarcasm_cue_tags(text)
    print(f"--- {title} ---")
    print(f"raw:       {text}")
    if collapsed != text.strip():
        print(f"collapsed: {collapsed}")
    print(f"whitespace ({len(white):2d}): {white}")
    print(f"tweet-ish ({len(tweet):2d}): {tweet}")
    if cues:
        print(f"cue tags:  {cues}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "test", "subtest"),
        default="test",
        help="Which official split to pull curated rows from.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip the dataset files and only print the built-in examples.",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        help="Optional extra tweet strings to tokenize.",
    )
    args = parser.parse_args(argv)

    print(
        "Comma collapse matches data_utils.ReadOpen. "
        "The tweet-ish tokenizer is a documented stand-in for "
        "nltk.TweetTokenizer — not a bitwise clone.\n"
    )

    for i, text in enumerate(SYNTHETIC):
        show_one(f"synthetic[{i}]", text)

    if not args.synthetic_only:
        sentences, labels = load_split(args.split)
        for idx in CURATED[args.split]:
            if idx >= len(sentences):
                continue
            show_one(f"{args.split}[{idx}] label={int(labels[idx])}", sentences[idx])

    for i, text in enumerate(args.texts):
        show_one(f"cli[{i}]", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
