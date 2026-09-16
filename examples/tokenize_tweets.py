"""Show the example tokenizer next to the raw tweet line."""

from __future__ import annotations

import argparse
import sys

from .dataset_io import load_split
from .tokenize import tokenize_tweet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "subtest"), default="subtest")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--label",
        choices=("any", "sarc", "lit"),
        default="any",
        help="filter printed rows by gold label",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = load_split(args.split)
    printed = 0
    skipped = 0
    want = {"sarc": 1, "lit": 0}.get(args.label)
    for sentence, label in split.pairs():
        if want is not None and label != want:
            continue
        if skipped < args.offset:
            skipped += 1
            continue
        tokens = tokenize_tweet(sentence)
        tag = "sarc" if label == 1 else "lit "
        print(f"[{tag}] {sentence}")
        print(f"       {tokens}")
        printed += 1
        if printed >= args.limit:
            break
    if printed == 0:
        print("no tweets matched", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
