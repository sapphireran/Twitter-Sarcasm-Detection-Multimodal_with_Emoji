#!/usr/bin/env python3
"""Print split sizes, label balance, length stats, and emoji / hashtag rates.

This is the lightweight stand-in for opening the CSVs in a notebook. It
does not need GloVe, TensorFlow, or pandas.

Usage (from the repo root):

    python3 examples/explore_dataset.py
    python3 examples/explore_dataset.py --split test --samples 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset import SPLITS, iter_splits, load_split, split_summary
from examples.lib.tokenize import extract_emojis, extract_hashtags, tokenize_tweet


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{100.0 * n / d:5.1f}%"


def summarize_split(name: str, samples: int) -> dict[str, object]:
    split = load_split(name)
    summary = split_summary(split)
    emoji_by_label = Counter()
    hashtag_by_label = Counter()
    token_lengths: list[int] = []
    tag_counts = Counter()
    emoji_counts = Counter()

    for text, label in zip(split.texts, split.labels):
        tokens = tokenize_tweet(text)
        token_lengths.append(len(tokens))
        emojis = extract_emojis(text)
        tags = extract_hashtags(text)
        if emojis:
            emoji_by_label[label] += 1
            emoji_counts.update(emojis)
        if tags:
            hashtag_by_label[label] += 1
            tag_counts.update(tags)

    n = len(split)
    n_sarc = summary["n_sarcastic"]
    n_non = summary["n_non_sarcastic"]
    summary.update(
        {
            "mean_tokens": sum(token_lengths) / n if n else 0.0,
            "emoji_rate_sarcastic": (emoji_by_label[1] / n_sarc) if n_sarc else 0.0,
            "emoji_rate_non_sarcastic": (emoji_by_label[0] / n_non) if n_non else 0.0,
            "hashtag_rate_sarcastic": (hashtag_by_label[1] / n_sarc) if n_sarc else 0.0,
            "hashtag_rate_non_sarcastic": (hashtag_by_label[0] / n_non) if n_non else 0.0,
            "top_hashtags": tag_counts.most_common(8),
            "top_emojis": emoji_counts.most_common(8),
        }
    )

    print(f"## {name}")
    print(f"  rows              {n}")
    print(
        f"  sarcastic         {n_sarc}  ({_pct(n_sarc, n)})   "
        f"non-sarcastic {n_non}  ({_pct(n_non, n)})"
    )
    print(
        f"  mean / median / max chars   "
        f"{summary['mean_chars']:.1f} / {summary['median_chars']:.0f} / {summary['max_chars']}"
    )
    print(f"  mean tokens       {summary['mean_tokens']:.1f}")
    print(
        f"  tweets with emoji     sarc {_pct(emoji_by_label[1], n_sarc)}   "
        f"non {_pct(emoji_by_label[0], n_non)}"
    )
    print(
        f"  tweets with hashtag   sarc {_pct(hashtag_by_label[1], n_sarc)}   "
        f"non {_pct(hashtag_by_label[0], n_non)}"
    )
    print("  top hashtags     " + ", ".join(f"#{t} ({c})" for t, c in summary["top_hashtags"]))
    print("  top emoji        " + ", ".join(f"{e} ({c})" for e, c in summary["top_emojis"]))

    if samples > 0:
        print("  samples")
        shown = 0
        for text, label in zip(split.texts, split.labels):
            if shown >= samples:
                break
            preview = text.replace("\n", " ")
            if len(preview) > 110:
                preview = preview[:107] + "..."
            print(f"    [{label}] {preview}")
            shown += 1
    print()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        choices=list(SPLITS),
        help="Limit to one or more splits (default: all).",
    )
    parser.add_argument("--samples", type=int, default=2, help="Preview rows per split.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also dump the numeric summary as JSON on stdout after the text report.",
    )
    args = parser.parse_args(argv)
    names = tuple(args.split) if args.split else SPLITS
    print("Twitter sarcasm dump — split overview")
    print("Source files live in dataset/. Labels: 0 = not sarcastic, 1 = sarcastic.\n")
    reports = [summarize_split(name, args.samples) for name in names]
    if args.json:
        print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
