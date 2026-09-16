#!/usr/bin/env python3
"""Show the ReadOpen comma step plus tokenization on real test tweets.

Usage:

    python3 examples/tokenize_tweets.py
    python3 examples/tokenize_tweets.py --limit 6 --label 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset_io import load_split  # noqa: E402
from examples.lib.tweet_tokenize import (  # noqa: E402
    extract_emoji,
    extract_hashtags,
    is_elongated,
    strip_commas_like_readopen,
    tokenize_tweet,
)

# Hand-picked test rows that exercise commas, emoji, tags, elongation.
FEATURED = (
    'I loovee when people text back ... 😒 #sarcastictweet',
    'Oh how I love getting home from work at 3am and my house being dirty #not',
    '"So many useless classes , great to be student"',
    "Don't you love it when your parents are Pissed because you were gonna study after bubble soccer ! #IKnowIDo #not 😃 🔫",
    'Want to have someone to speak to I\'m so bored 😭',
    'i just imagined you dancing like this',
)


def show(sentence: str, label: int | None = None) -> None:
    cleaned = strip_commas_like_readopen(sentence)
    tokens = tokenize_tweet(sentence)
    prefix = "" if label is None else f"[{label}] "
    print(f"{prefix}raw:     {sentence}")
    if cleaned != sentence.strip():
        print(f"       commas→  {cleaned}")
    print(f"       tokens:  {tokens}")
    tags = extract_hashtags(tokens)
    emoji = extract_emoji(tokens)
    elong = [tok for tok in tokens if is_elongated(tok)]
    extras = []
    if tags:
        extras.append("tags=" + ",".join(tags))
    if emoji:
        extras.append("emoji=" + ",".join(emoji))
    if elong:
        extras.append("elong=" + ",".join(elong))
    if extras:
        print("       cues:    " + "  ".join(extras))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument(
        "--from-split",
        choices=("featured", "test", "train", "subtest"),
        default="featured",
        help="featured = the hand-picked rows documented in docs/",
    )
    parser.add_argument("--label", type=int, choices=(0, 1), default=None)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args(argv)

    if args.from_split == "featured":
        for sentence in FEATURED[: args.limit]:
            show(sentence)
        return 0

    split = load_split(args.from_split, args.dataset_dir)
    shown = 0
    for sentence, label in split.labeled():
        if args.label is not None and label != args.label:
            continue
        show(sentence, label=label)
        shown += 1
        if shown >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
