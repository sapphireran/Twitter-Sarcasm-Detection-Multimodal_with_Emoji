"""Pretty-print the frozen 2023 metric cards."""

from __future__ import annotations

import argparse
import sys

from .results_catalog import (
    ALL_CARDS,
    ascii_table,
    best_card,
    emoji_gain,
    markdown_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        choices=("accuracy", "f1", "precision", "recall"),
        default="accuracy",
    )
    parser.add_argument(
        "--format",
        choices=("ascii", "markdown"),
        default="ascii",
        dest="table_format",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="print every recorded metric, not just --metric",
    )
    return parser


def _cards_for(metric: str):
    return [card for card in ALL_CARDS if getattr(card, metric) is not None]


def _print_table(metric: str, table_format: str) -> None:
    cards = _cards_for(metric)
    print(f"## {metric}")
    if table_format == "markdown":
        print(markdown_table(cards, metric))
    else:
        print(ascii_table(cards, metric))
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = ("accuracy", "f1", "precision", "recall") if args.all_metrics else (args.metric,)
    print("June 2023 recorded results (not recomputed).")
    print("Column order: test word | test word+emoji | subtest word | subtest word+emoji")
    print()
    for metric in metrics:
        _print_table(metric, args.table_format)

    print("Emoji-channel accuracy gain (multi-modal minus single-modal)")
    for card in ALL_CARDS:
        gain = emoji_gain(card, "accuracy")
        print(
            f"  {card.model:<22} "
            f"test {gain['test_gain']:+.4f}   "
            f"subtest {gain['subtest_gain']:+.4f}"
        )
    winner = best_card(_cards_for("accuracy"), "accuracy", 1)
    print()
    print(f"Best test word+emoji accuracy: {winner.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
