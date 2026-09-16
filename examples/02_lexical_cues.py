#!/usr/bin/env python3
"""Measure hashtag leakage and a tiny tag-only rule baseline."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.io import SPLITS, load_split
from examples.lib.tokenize import SARCASM_TAGS, find_emoji, find_hashtags, has_sarcasm_tag


def _precision_recall(hits: list[int], labels: list[int]) -> tuple[float, float, int]:
    """Precision/recall of a boolean rule against the sarcastic class."""
    tp = sum(1 for hit, label in zip(hits, labels) if hit and label == 1)
    fp = sum(1 for hit, label in zip(hits, labels) if hit and label == 0)
    fn = sum(1 for hit, label in zip(hits, labels) if (not hit) and label == 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, tp + fp


def tag_table(rows: list[tuple[str, int]]) -> None:
    labels = [label for _, label in rows]
    print(f"{'cue':<22} {'coverage':>9} {'sarcastic':>10} {'sincere':>9} {'prec':>6} {'rec':>6}")
    for tag in SARCASM_TAGS:
        hits = [int(tag in find_hashtags(text)) for text, _ in rows]
        prec, rec, covered = _precision_recall(hits, labels)
        sarcastic = sum(1 for hit, label in zip(hits, labels) if hit and label == 1)
        sincere = covered - sarcastic
        print(
            f"{tag:<22} {covered / len(rows):>8.1%} {sarcastic:>10} {sincere:>9} "
            f"{prec:>6.3f} {rec:>6.3f}"
        )

    emoji_hits = [int(bool(find_emoji(text))) for text, _ in rows]
    prec, rec, covered = _precision_recall(emoji_hits, labels)
    sarcastic = sum(1 for hit, label in zip(emoji_hits, labels) if hit and label == 1)
    print(
        f"{'any emoji':<22} {covered / len(rows):>8.1%} {sarcastic:>10} "
        f"{covered - sarcastic:>9} {prec:>6.3f} {rec:>6.3f}"
    )

    rule_hits = [int(has_sarcasm_tag(text)) for text, _ in rows]
    prec, rec, covered = _precision_recall(rule_hits, labels)
    acc = sum(1 for hit, label in zip(rule_hits, labels) if hit == label) / len(rows)
    print(
        f"{'tag rule (any of 5)':<22} {covered / len(rows):>8.1%} "
        f"{sum(1 for h, y in zip(rule_hits, labels) if h and y == 1):>10} "
        f"{sum(1 for h, y in zip(rule_hits, labels) if h and y == 0):>9} "
        f"{prec:>6.3f} {rec:>6.3f}"
    )
    print(f"  tag-rule accuracy treating 'no tag' as sincere: {acc:.3f}")


def frequent_tags(rows: list[tuple[str, int]], k: int = 8) -> None:
    pos: Counter[str] = Counter()
    neg: Counter[str] = Counter()
    for text, label in rows:
        tags = find_hashtags(text)
        (pos if label == 1 else neg).update(tags)
    combined = pos + neg
    print("  most frequent hashtags (sarcastic / sincere):")
    for tag, _ in combined.most_common(k):
        print(f"    {tag:<22} {pos[tag]:>5} / {neg[tag]:<5}")


def main() -> None:
    print(
        "Lexical cues on the shipped splits.\n"
        "A tag such as #not is a near-perfect sarcasm feature on the test set;\n"
        "treat neural scores with that ceiling in mind.\n"
    )
    for name in SPLITS:
        rows = load_split(name)
        print(f"## {name}  (n={len(rows)})")
        tag_table(rows)
        if name == "train":
            frequent_tags(rows)
        print()
    print("See docs/dataset.md for the same counts written out in prose.")


if __name__ == "__main__":
    main()
