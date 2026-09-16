#!/usr/bin/env python3
"""Print split sizes, cue rates, and the subtest ⊂ test relationship."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarcasm_lib.emoji import has_emoji
from sarcasm_lib.io import load_all_splits
from sarcasm_lib.stats import compute_split_stats, stats_as_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable stats instead of the text report",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="sarcastic examples to print per split (text mode only)",
    )
    return parser


def print_report(splits: dict, example_count: int) -> None:
    test_texts = set(splits["test"].texts)
    sub_in_test = sum(text in test_texts for text in splits["subtest"].texts)
    emoji_test = sum(has_emoji(text) for text in splits["test"].texts)

    print("Twitter sarcasm splits")
    print("======================")
    print(
        f"subtest tweets also in test: {sub_in_test}/{len(splits['subtest'])} "
        f"(emoji-bearing test tweets: {emoji_test})"
    )
    print()

    for name, split in splits.items():
        stats = compute_split_stats(split)
        payload = stats_as_dict(stats)
        print(f"## {name}")
        print(
            f"n={payload['n']}  sarcastic={payload['sarcastic']} "
            f"({payload['sarcasm_rate']:.1%})  "
            f"emoji={payload['with_emoji']}  "
            f"marker hashtags={payload['with_marker_hashtag']} "
            f"(precision={payload['marker_precision']:.3f})"
        )
        print(f"mean tokens={payload['mean_tokens']}  mean emoji={payload['mean_emoji']}")
        print("top hashtags:", ", ".join(f"{tag} {n}" for tag, n in payload["top_hashtags"][:8]))
        print(
            "top emoji:",
            ", ".join(f"{glyph} {n}" for glyph, n in payload["top_emoji"][:8]) or "(none)",
        )
        printed = 0
        for text, label in split.pairs():
            if label == 1:
                snippet = " ".join(text.split())
                print(f"  sarcastic ex: {snippet[:160]}")
                printed += 1
                if printed >= example_count:
                    break
        print()


def main() -> None:
    args = build_parser().parse_args()
    splits = load_all_splits()
    if args.json:
        blob = {name: stats_as_dict(compute_split_stats(split)) for name, split in splits.items()}
        print(json.dumps(blob, ensure_ascii=False, indent=2))
        return
    print_report(splits, args.examples)


if __name__ == "__main__":
    main()
