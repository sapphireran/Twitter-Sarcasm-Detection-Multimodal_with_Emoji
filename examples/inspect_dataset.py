#!/usr/bin/env python3
"""Print split counts, class balance, and surface-cue rates.

Usage:
    python3 examples/inspect_dataset.py
    python3 examples/inspect_dataset.py --faithful
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from sarcasm_lib.dataset import SPLIT_NAMES, load_all_splits, word_count
from sarcasm_lib.features import summarize_cues
from sarcasm_lib.tokenize import extract_emoji, extract_hashtags


def _pct(part: int, whole: int) -> str:
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:5.1f}%"


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def describe_split(name: str, split, *, examples: int) -> None:
    cues = summarize_cues(split)
    lengths = [word_count(sentence) for sentence in split.sentences]
    print(f"## {name}")
    print(f"rows                 {cues.n}")
    print(
        f"class 0 / class 1    {cues.n_neg} ({_pct(cues.n_neg, cues.n)})  /  "
        f"{cues.n_pos} ({_pct(cues.n_pos, cues.n)})"
    )
    print(
        f"words mean/median/max {_mean(lengths):.2f} / {_median(lengths):.1f} / {max(lengths) if lengths else 0}"
    )
    print(
        f"mean words by class   0={cues.mean_words_neg:.2f}  1={cues.mean_words_pos:.2f}"
    )
    print(
        f"emoji tweets          {cues.emoji_neg + cues.emoji_pos} "
        f"({_pct(cues.emoji_neg + cues.emoji_pos, cues.n)})  "
        f"[0: {cues.emoji_neg}  1: {cues.emoji_pos}]"
    )
    print(
        f"hashtag tweets        {cues.hashtag_neg + cues.hashtag_pos} "
        f"({_pct(cues.hashtag_neg + cues.hashtag_pos, cues.n)})  "
        f"[0: {cues.hashtag_neg}  1: {cues.hashtag_pos}]"
    )
    print(
        f"#not tweets           {cues.not_tag_neg + cues.not_tag_pos}  "
        f"[0: {cues.not_tag_neg}  1: {cues.not_tag_pos}]"
    )
    print(
        f"<user> tweets         {cues.user_neg + cues.user_pos}  "
        f"[0: {cues.user_neg}  1: {cues.user_pos}]"
    )
    print("top emoji class 0    ", cues.emoji_counts_neg.most_common(8))
    print("top emoji class 1    ", cues.emoji_counts_pos.most_common(8))
    print("top hashtag class 0  ", cues.hashtag_counts_neg.most_common(6))
    print("top hashtag class 1  ", cues.hashtag_counts_pos.most_common(6))
    if examples:
        print("examples class 0:")
        for sentence in list(split.labeled(0))[:examples]:
            print(f"  · {sentence[:160]}")
        print("examples class 1:")
        for sentence in list(split.labeled(1))[:examples]:
            print(f"  · {sentence[:160]}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--faithful",
        action="store_true",
        help="replay data_utils.ReadOpen comma-stripping",
    )
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_NAMES),
        choices=SPLIT_NAMES,
    )
    args = parser.parse_args(argv)

    splits = load_all_splits(faithful=args.faithful, names=args.splits)
    print("# Dataset inspection")
    print(f"faithful_readopen = {args.faithful}")
    print()
    for name in args.splits:
        describe_split(name, splits[name], examples=args.examples)

    train = splits.get("train")
    if train is not None:
        empty_emoji = sum(
            1
            for sentence, label in train
            if label == 1 and not extract_emoji(sentence) and not extract_hashtags(sentence)
        )
        print(
            "train sarcastic tweets with neither emoji nor hashtag: "
            f"{empty_emoji} / {train.n_positive}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
