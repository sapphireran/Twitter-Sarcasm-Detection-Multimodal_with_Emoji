#!/usr/bin/env python3
"""Rank emoji and hashtags by informative log-odds with the sarcastic class.

Usage:
    python3 examples/emoji_signals.py
    python3 examples/emoji_signals.py --split test --top 12
    python3 examples/emoji_signals.py --markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from sarcasm_lib.dataset import SPLIT_NAMES, load_split
from sarcasm_lib.features import (
    count_tokens,
    format_association_table,
    informative_log_odds,
)
from sarcasm_lib.tokenize import extract_emoji, extract_hashtags


def _print_block(title: str, rows, *, top: int, markdown: bool) -> None:
    print(f"### {title}")
    if markdown:
        print()
        print("Most sarcastic-leaning:")
        print(format_association_table(rows, limit=top, positive=True))
        print()
        print("Most non-sarcastic-leaning:")
        print(format_association_table(rows, limit=top, positive=False))
        print()
        return
    print(f"{'token':<24} {'c0':>6} {'c1':>6} {'log-odds':>10} {'z':>8}")
    print("-" * 58)
    print("sarcastic-leaning")
    for row in rows[:top]:
        print(
            f"{row.token:<24} {row.count_neg:6d} {row.count_pos:6d} "
            f"{row.log_odds:+10.3f} {row.z_score:+8.2f}"
        )
    print("non-sarcastic-leaning")
    for row in list(reversed(rows))[:top]:
        print(
            f"{row.token:<24} {row.count_neg:6d} {row.count_pos:6d} "
            f"{row.log_odds:+10.3f} {row.z_score:+8.2f}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=SPLIT_NAMES)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--faithful", action="store_true")
    args = parser.parse_args(argv)

    split = load_split(args.split, faithful=args.faithful)
    pos_texts = list(split.labeled(1))
    neg_texts = list(split.labeled(0))

    emoji_rows = informative_log_odds(
        count_tokens(neg_texts, extract_emoji),
        count_tokens(pos_texts, extract_emoji),
        min_count=args.min_count,
    )
    tag_rows = informative_log_odds(
        count_tokens(neg_texts, extract_hashtags),
        count_tokens(pos_texts, extract_hashtags),
        min_count=args.min_count,
    )

    print(f"# Emoji / hashtag log-odds on `{args.split}` (n={len(split)})")
    print(
        "Positive log-odds = more common in sarcastic tweets after "
        "add-one Dirichlet smoothing (Monroe et al. 2008)."
    )
    print()
    _print_block("Emoji", emoji_rows, top=args.top, markdown=args.markdown)
    _print_block("Hashtags", tag_rows, top=args.top, markdown=args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
