#!/usr/bin/env python3
"""Measure how surface sarcasm cues line up with the committed labels."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sarcasm_lab.cues import extract_cues
from sarcasm_lab.io import Split, load_all_splits
from sarcasm_lab.tables import format_table


def _rate(flag: list[bool]) -> float:
    return 100.0 * sum(flag) / len(flag) if flag else 0.0


def _group_rates(split: Split, attr: str) -> tuple[str, str, str]:
    by_label: dict[int, list[bool]] = {0: [], 1: []}
    for text, tokens, label in zip(split.texts, split.tokens, split.labels):
        cues = extract_cues(text, tokens)
        by_label[label].append(bool(getattr(cues, attr)))
    overall = _rate(by_label[0] + by_label[1])
    return f"{overall:.1f}%", f"{_rate(by_label[0]):.1f}%", f"{_rate(by_label[1]):.1f}%"


def _top_hashtags(split: Split, n: int = 8) -> list[tuple[str, int, float]]:
    counts: Counter[str] = Counter()
    sarcastic: Counter[str] = Counter()
    for text, tokens, label in zip(split.texts, split.tokens, split.labels):
        cues = extract_cues(text, tokens)
        seen = set(cues.hashtags)
        for tag in seen:
            counts[tag] += 1
            if label == 1:
                sarcastic[tag] += 1
    rows = []
    for tag, total in counts.most_common(n):
        rows.append((tag, total, 100.0 * sarcastic[tag] / total))
    return rows


def _examples(split: Split, want_sarcasm: bool, limit: int = 3) -> list[str]:
    out = []
    for text, tokens, label in zip(split.texts, split.tokens, split.labels):
        if bool(label) != want_sarcasm:
            continue
        cues = extract_cues(text, tokens)
        if want_sarcasm and not (cues.has_sarcasm_hashtag or cues.has_negative_emoji):
            continue
        if not want_sarcasm and (cues.has_sarcasm_hashtag or cues.n_emoji):
            continue
        snippet = text.replace("\n", " ")
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        out.append(snippet)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    splits = load_all_splits(tokenize=True)
    attrs = [
        ("has_sarcasm_hashtag", "sarcasm hashtag"),
        ("has_negative_emoji", "neg. emoji"),
        ("has_positive_stem", "positive stem"),
    ]
    print("Share of tweets with each cue (overall / not-sarcastic / sarcastic)")
    rows = []
    for name in ("train", "test", "subtest"):
        split = splits[name]
        for attr, label in attrs:
            overall, neg, pos = _group_rates(split, attr)
            rows.append([name, label, overall, neg, pos])
    print(
        format_table(
            ["split", "cue", "all", "label0", "label1"],
            rows,
            right_align={2, 3, 4},
        )
    )

    print()
    print("Most common hashtags on the training set (and % that are sarcastic)")
    hash_rows = [
        [tag, str(total), f"{pct:.1f}%"]
        for tag, total, pct in _top_hashtags(splits["train"], n=10)
    ]
    print(format_table(["hashtag", "tweets", "% sarcastic"], hash_rows, right_align={1, 2}))

    print()
    print("Most common hashtags on the subtest")
    sub_rows = [
        [tag, str(total), f"{pct:.1f}%"]
        for tag, total, pct in _top_hashtags(splits["subtest"], n=8)
    ]
    print(format_table(["hashtag", "tweets", "% sarcastic"], sub_rows, right_align={1, 2}))

    print()
    print("Sample sarcastic subtest lines that carry a cue")
    for line in _examples(splits["subtest"], want_sarcasm=True):
        print(f"  · {line}")
    print("Sample non-sarcastic train lines without those cues")
    for line in _examples(splits["train"], want_sarcasm=False):
        print(f"  · {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
