#!/usr/bin/env python3
"""Print split sizes, label balance, and basic length statistics."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sarcasm_lab.io import load_all_splits
from sarcasm_lab.tables import format_table
from sarcasm_lab.tokenize import is_emoji_token, is_hashtag


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def main() -> int:
    splits = load_all_splits(tokenize=True)
    rows = []
    detail_rows = []
    for name in ("train", "test", "subtest"):
        split = splits[name]
        counts = split.label_counts()
        lengths = [len(toks) for toks in split.tokens]
        char_lengths = [len(text) for text in split.texts]
        n_hash = sum(1 for toks in split.tokens if any(is_hashtag(t) for t in toks))
        n_emoji = sum(1 for toks in split.tokens if any(is_emoji_token(t) for t in toks))
        n_user = sum(1 for toks in split.tokens if "<user>" in toks)
        rows.append(
            [
                name,
                str(len(split)),
                str(counts[0]),
                str(counts[1]),
                f"{100 * split.sarcastic_rate():.1f}%",
                f"{_mean(lengths):.1f}",
                f"{_percentile(lengths, 0.5):.0f}",
                f"{_percentile(lengths, 0.95):.0f}",
                str(max(lengths) if lengths else 0),
            ]
        )
        detail_rows.append(
            [
                name,
                f"{100 * n_hash / len(split):.1f}%",
                f"{100 * n_emoji / len(split):.1f}%",
                f"{100 * n_user / len(split):.1f}%",
                f"{_mean(char_lengths):.1f}",
            ]
        )

    print("Splits (labels 0 = not sarcastic, 1 = sarcastic)")
    print(
        format_table(
            ["split", "n", "label0", "label1", "sarcasm", "mean tok", "p50", "p95", "max"],
            rows,
            right_align={1, 2, 3, 4, 5, 6, 7, 8},
        )
    )
    print()
    print("How often a tweet contains at least one hashtag / emoji / <user>")
    print(
        format_table(
            ["split", "hashtag", "emoji", "mention", "mean chars"],
            detail_rows,
            right_align={1, 2, 3, 4},
        )
    )
    print()
    print("Notes")
    print("- test is balanced by construction (1000 / 1000).")
    print("- subtest is smaller, sarcasm-heavy, and denser in emoji/hashtags.")
    train_max = max(len(toks) for toks in splits["train"].tokens)
    print(f"- this tokenizer's train max length: {train_max} (Keras / NLTK used 78).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
