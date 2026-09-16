"""Print split sizes, label layout, and surface-cue rates."""

from __future__ import annotations

import argparse
import sys

from .dataset_io import (
    assert_subtest_is_test_emoji,
    format_summary,
    load_split,
    summarize_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "test", "subtest", "all"),
        default="all",
    )
    parser.add_argument("--examples", type=int, default=3, help="tweets to print per split")
    parser.add_argument(
        "--check-subtest",
        action="store_true",
        default=True,
        help="assert subtest == emoji subset of test (default on)",
    )
    parser.add_argument(
        "--no-check-subtest",
        action="store_false",
        dest="check_subtest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = ("train", "test", "subtest") if args.split == "all" else (args.split,)
    for name in names:
        split = load_split(name)
        summary = summarize_split(split)
        print(format_summary(summary))
        if args.examples > 0:
            print("examples")
            for sentence, label in split.head(args.examples):
                tag = "sarc" if label == 1 else "lit "
                print(f"  [{tag}] {sentence}")
        print()

    if args.check_subtest and (args.split in {"all", "subtest", "test"}):
        assert_subtest_is_test_emoji(load_split("test"), load_split("subtest"))
        print("check: subtest is the emoji-bearing subset of test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
