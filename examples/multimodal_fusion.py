"""Show 200-d vs 400-d early fusion using tiny synthetic tables."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .fusion import pool_tweet, toy_tables


DEFAULT_TWEETS = (
    "I love walking to school",
    "I love walking to school 😒",
    "I love walking to school #not 😒",
    "Great day 😃",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4, help="synthetic embedding width")
    parser.add_argument(
        "--tweet",
        action="append",
        dest="tweets",
        help="override the demo tweets; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    word_table, emoji_table = toy_tables(dim=args.dim)
    tweets = args.tweets or list(DEFAULT_TWEETS)
    print(f"toy GloVe rows: {sorted(word_table)}  dim={args.dim}")
    print(f"toy emoji rows: {sorted(emoji_table)}")
    print()
    print("Classical path: mean word vector, then concat mean emoji vector.")
    print("A tweet with no in-vocabulary emoji gets a zero emoji channel.")
    print()
    for tweet in tweets:
        pooled = pool_tweet(tweet, word_table, emoji_table, args.dim)
        emoji_norm = float(np.linalg.norm(pooled.emoji_mean))
        print(f"tweet     {tweet}")
        print(f"tokens    {pooled.tokens}")
        print(f"word hits {pooled.word_hits}")
        print(f"emoji hits {pooled.emoji_hits}")
        print(f"word mean {np.round(pooled.word_mean, 3)}   shape {pooled.word_mean.shape}")
        print(
            f"emoji mean {np.round(pooled.emoji_mean, 3)}  "
            f"L2={emoji_norm:.3f}   fused shape {pooled.concatenated.shape}"
        )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
