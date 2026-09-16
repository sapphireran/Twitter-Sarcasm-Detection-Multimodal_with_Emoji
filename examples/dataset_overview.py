#!/usr/bin/env python3
"""Print split sizes, label balance, and emoji rates for the project CSVs.

Run from the repository root:

    python examples/dataset_overview.py
    python examples/dataset_overview.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dataset_io import SPLIT_FILES, iter_split
from examples.emoji import extract_emojis, has_emoji
from examples.tokenize import tokenize_tweet


def summarize_split(name: str) -> dict:
    texts, labels = iter_split(name)
    if len(texts) != len(labels):
        raise RuntimeError(
            f"{name}: {len(texts)} sentences vs {len(labels)} labels"
        )
    label_counts = Counter(labels)
    emoji_flags = [has_emoji(text) for text in texts]
    emoji_by_label = Counter()
    cue_not = 0
    cue_sarcasm = 0
    token_lengths = []
    emoji_glyphs: Counter[str] = Counter()
    for text, label, flagged in zip(texts, labels, emoji_flags):
        tokens = tokenize_tweet(text)
        token_lengths.append(len(tokens))
        if flagged:
            emoji_by_label[label] += 1
            emoji_glyphs.update(extract_emojis(text))
        lowered = {tok.lower() for tok in tokens}
        if "#not" in lowered:
            cue_not += 1
        if lowered & {"#sarcasm", "#sarcastic", "#sarcastictweet"}:
            cue_sarcasm += 1

    n = len(texts)
    return {
        "split": name,
        "n": n,
        "n_non_sarcastic": int(label_counts.get(0, 0)),
        "n_sarcastic": int(label_counts.get(1, 0)),
        "sarcastic_rate": round(label_counts.get(1, 0) / n, 4) if n else 0.0,
        "n_with_emoji": int(sum(emoji_flags)),
        "emoji_rate": round(sum(emoji_flags) / n, 4) if n else 0.0,
        "emoji_in_non_sarcastic": int(emoji_by_label.get(0, 0)),
        "emoji_in_sarcastic": int(emoji_by_label.get(1, 0)),
        "n_with_hash_not": cue_not,
        "n_with_sarcasm_hashtag": cue_sarcasm,
        "mean_tokens": round(sum(token_lengths) / n, 2) if n else 0.0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "top_emoji": emoji_glyphs.most_common(8),
        "sentence_file": str(SPLIT_FILES[name][0].relative_to(SPLIT_FILES[name][0].parents[1])),
        "label_file": str(SPLIT_FILES[name][1].relative_to(SPLIT_FILES[name][1].parents[1])),
    }


def render_table(rows: list[dict]) -> str:
    headers = [
        "split",
        "n",
        "non-sarc",
        "sarc",
        "sarc%",
        "emoji",
        "emoji%",
        "mean tok",
        "#not",
        "sarc-tag",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["split"],
                    str(row["n"]),
                    str(row["n_non_sarcastic"]),
                    str(row["n_sarcastic"]),
                    f"{100 * row['sarcastic_rate']:.1f}",
                    str(row["n_with_emoji"]),
                    f"{100 * row['emoji_rate']:.1f}",
                    f"{row['mean_tokens']:.2f}",
                    str(row["n_with_hash_not"]),
                    str(row["n_with_sarcasm_hashtag"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable stats")
    args = parser.parse_args(argv)

    rows = [summarize_split(name) for name in ("train", "test", "subtest")]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    print("Twitter sarcasm corpus — split overview")
    print()
    print(render_table(rows))
    print()
    print("Notes")
    print("- Label 1 is sarcastic, label 0 is not.")
    print("- `subtest` is the emoji-bearing slice of `test` (same tweets, 278 rows).")
    print("- `#not` / sarcasm hashtags are surface cues, not the model features.")
    print()
    for row in rows:
        top = ", ".join(f"{glyph}×{count}" for glyph, count in row["top_emoji"]) or "(none)"
        print(f"Top emoji in {row['split']}: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
