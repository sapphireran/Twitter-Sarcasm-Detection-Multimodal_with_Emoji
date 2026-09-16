#!/usr/bin/env python3
"""Count surface sarcasm cues by split and label.

The classifiers do not get these counts as explicit features. They see
GloVe / emoji2vec means (baselines) or a BiLSTM over the token ids
(deep model). The counts are still the easiest way to see *why* a
held-out emoji slice (``subtest``) behaves differently from the full
test set.

Run from the repository root:

    python examples/sarcasm_cues.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dataset_io import iter_split
from examples.emoji import has_emoji
from examples.tokenize import tokenize_tweet

SARCASM_TAGS = {"#not", "#sarcasm", "#sarcastic", "#sarcastictweet", "#yeahright"}


def cue_row(text: str, label: int) -> dict:
    tokens = [tok.lower() for tok in tokenize_tweet(text)]
    tags = [tok for tok in tokens if tok in SARCASM_TAGS]
    return {
        "label": label,
        "has_emoji": has_emoji(text),
        "has_sarc_tag": bool(tags),
        "tags": tags,
        "n_tokens": len(tokens),
    }


def summarize(name: str) -> dict:
    texts, labels = iter_split(name)
    by_label = {
        0: Counter(emoji=0, tag=0, both=0, neither=0, n=0),
        1: Counter(emoji=0, tag=0, both=0, neither=0, n=0),
    }
    tag_freq = Counter()
    for text, label in zip(texts, labels):
        row = cue_row(text, label)
        bucket = by_label[label]
        bucket["n"] += 1
        if row["has_emoji"]:
            bucket["emoji"] += 1
        if row["has_sarc_tag"]:
            bucket["tag"] += 1
            tag_freq.update(row["tags"])
        if row["has_emoji"] and row["has_sarc_tag"]:
            bucket["both"] += 1
        if not row["has_emoji"] and not row["has_sarc_tag"]:
            bucket["neither"] += 1
    return {
        "split": name,
        "by_label": {str(k): dict(v) for k, v in by_label.items()},
        "tag_freq": tag_freq.most_common(),
    }


def render(summary: dict) -> str:
    lines = [f"## {summary['split']}", ""]
    lines.append("| label | n | emoji | sarc-tag | both | neither |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for label in ("0", "1"):
        row = summary["by_label"][label]
        name = "non-sarc" if label == "0" else "sarcastic"
        lines.append(
            f"| {label} ({name}) | {row['n']} | {row['emoji']} | "
            f"{row['tag']} | {row['both']} | {row['neither']} |"
        )
    lines.append("")
    if summary["tag_freq"]:
        pretty = ", ".join(f"{tag}×{count}" for tag, count in summary["tag_freq"][:8])
        lines.append(f"tag frequency: {pretty}")
    else:
        lines.append("tag frequency: (none of the tracked tags)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summaries = [summarize(name) for name in ("train", "test", "subtest")]
    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return 0
    print("Surface cues vs. gold sarcasm label")
    print()
    for summary in summaries:
        print(render(summary))
    print(
        "Reading the tables: multi-modal emoji vectors can only move the "
        "decision on rows where `emoji` is non-zero. That is ~14% of train/test "
        "and essentially all of subtest — which is why the emoji lift is "
        "larger on the subtest numbers in `docs/results.md`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
