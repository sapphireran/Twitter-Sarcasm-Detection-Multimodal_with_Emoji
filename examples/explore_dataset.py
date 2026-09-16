#!/usr/bin/env python3
"""Print split sizes, label balance, cue rates, and length histograms.

This is the runnable companion to docs/dataset.md. It only needs the CSV
files under dataset/ plus NumPy.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

# Allow `python examples/explore_dataset.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EMOJI_RE,
    HASHTAG_RE,
    SPLITS,
    dump_json,
    has_emoji,
    has_mention,
    iter_hashtags,
    load_split,
    sarcasm_cue_tags,
    whitespace_tokens,
)


def count_by_label(labels) -> dict:
    counter = collections.Counter(int(x) for x in labels)
    total = sum(counter.values())
    return {
        "0": counter.get(0, 0),
        "1": counter.get(1, 0),
        "n": total,
        "frac_0": counter.get(0, 0) / total,
        "frac_1": counter.get(1, 0) / total,
    }


def length_stats(sentences) -> dict:
    lengths = [len(whitespace_tokens(s)) for s in sentences]
    return {
        "min": min(lengths),
        "median": float(statistics.median(lengths)),
        "mean": float(statistics.mean(lengths)),
        "max": max(lengths),
    }


def surface_stats(sentences, labels) -> dict:
    n = len(sentences)
    emoji_n = sum(1 for s in sentences if has_emoji(s))
    hash_n = sum(1 for s in sentences if HASHTAG_RE.search(s))
    mention_n = sum(1 for s in sentences if has_mention(s))
    cue_n = sum(1 for s in sentences if sarcasm_cue_tags(s))
    by_label = {}
    for lab in (0, 1):
        subset = [s for s, y in zip(sentences, labels) if int(y) == lab]
        by_label[str(lab)] = {
            "n": len(subset),
            "emoji": sum(1 for s in subset if has_emoji(s)),
            "emoji_rate": (sum(1 for s in subset if has_emoji(s)) / len(subset))
            if subset
            else 0.0,
            "cue": sum(1 for s in subset if sarcasm_cue_tags(s)),
        }
    tags = collections.Counter()
    for s in sentences:
        tags.update(iter_hashtags(s))
    return {
        "with_emoji": emoji_n,
        "with_emoji_frac": emoji_n / n,
        "with_hashtag": hash_n,
        "with_hashtag_frac": hash_n / n,
        "with_mention": mention_n,
        "with_mention_frac": mention_n / n,
        "with_sarcasm_cue_tag": cue_n,
        "with_sarcasm_cue_tag_frac": cue_n / n,
        "by_label": by_label,
        "top_hashtags": tags.most_common(15),
    }


def length_histogram(sentences, width: int = 40) -> str:
    lengths = [len(whitespace_tokens(s)) for s in sentences]
    buckets = collections.Counter((length // 5) * 5 for length in lengths)
    max_count = max(buckets.values())
    lines = ["  tokens   count  bar"]
    for start in range(0, max(buckets) + 5, 5):
        count = buckets.get(start, 0)
        bar_len = 0 if max_count == 0 else int(round(width * count / max_count))
        lines.append(f"  {start:2d}-{start + 4:2d}  {count:6d}  {'#' * bar_len}")
    return "\n".join(lines)


def summarize_split(split: str) -> dict:
    sentences, labels = load_split(split)
    return {
        "split": split,
        "labels": count_by_label(labels),
        "length": length_stats(sentences),
        "surface": surface_stats(sentences, labels),
    }


def print_human(summary: dict) -> None:
    lab = summary["labels"]
    length = summary["length"]
    surface = summary["surface"]
    print(f"=== {summary['split']} ===")
    print(
        f"n={lab['n']}  label0={lab['0']} ({lab['frac_0']:.1%})  "
        f"label1={lab['1']} ({lab['frac_1']:.1%})"
    )
    print(
        f"whitespace tokens  min={length['min']}  "
        f"median={length['median']:.1f}  mean={length['mean']:.2f}  "
        f"max={length['max']}"
    )
    print(
        f"emoji={surface['with_emoji']} ({surface['with_emoji_frac']:.1%})  "
        f"hashtag={surface['with_hashtag']} ({surface['with_hashtag_frac']:.1%})  "
        f"mention={surface['with_mention']} ({surface['with_mention_frac']:.1%})  "
        f"cue-tag={surface['with_sarcasm_cue_tag']} "
        f"({surface['with_sarcasm_cue_tag_frac']:.1%})"
    )
    for lab_key, row in surface["by_label"].items():
        rate = row["emoji_rate"]
        print(
            f"  label {lab_key}: n={row['n']}  emoji={row['emoji']} "
            f"({rate:.1%})  cue-tag={row['cue']}"
        )
    print("  top hashtags:")
    for tag, count in surface["top_hashtags"]:
        print(f"    {count:5d}  {tag}")
    print("  length histogram (whitespace tokens):")
    sentences, _ = load_split(summary["split"])
    print(length_histogram(sentences))
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON object instead of the text report.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        action="append",
        help="Restrict to one or more splits. Default: all.",
    )
    args = parser.parse_args(argv)
    splits = args.split or list(SPLITS)
    summaries = [summarize_split(split) for split in splits]
    if args.json:
        print(dump_json({"splits": summaries}))
    else:
        for summary in summaries:
            print_human(summary)
        print(
            "Notes: subtest is the emoji-co-occurrence slice "
            "(almost every row has an emoji). "
            "Test/subtest mention rate is near zero while train still "
            "contains thousands of <user> tokens."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
