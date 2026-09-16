#!/usr/bin/env python3
"""Print a structured report of the three sarcasm splits.

Run from the repo root::

    python3 examples/01_explore_dataset.py
    python3 examples/01_explore_dataset.py --markdown > examples/output/dataset_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.emoji import emoji_code_points
from examples.common.io import iter_splits
from examples.common.tokenize import CUE_HASHTAGS, tokenize_tweet


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def summarize_split(split) -> dict:
    n = len(split)
    token_lens = [len(tokenize_tweet(text)) for text in split.texts]
    char_lens = [len(text) for text in split.texts]
    emoji_counts = [len(emoji_code_points(text)) for text in split.texts]
    has_emoji = sum(1 for c in emoji_counts if c > 0)
    has_hash = 0
    has_cue = 0
    user_ph = 0
    emoji_by_label = {0: 0, 1: 0}
    cue_by_label = {0: 0, 1: 0}
    for text, label in split.pairs():
        tokens = tokenize_tweet(text)
        if any(tok.startswith("#") for tok in tokens):
            has_hash += 1
        if any(tok in CUE_HASHTAGS for tok in tokens):
            has_cue += 1
            cue_by_label[label] += 1
        if "<user>" in tokens:
            user_ph += 1
        if emoji_code_points(text):
            emoji_by_label[label] += 1
    return {
        "name": split.name,
        "n": n,
        "n_literal": split.n_literal,
        "n_sarcastic": split.n_sarcastic,
        "literal_pct": round(_pct(split.n_literal, n), 2),
        "sarcastic_pct": round(_pct(split.n_sarcastic, n), 2),
        "tokens_mean": round(statistics.fmean(token_lens), 3),
        "tokens_median": statistics.median(token_lens),
        "tokens_max": max(token_lens),
        "chars_mean": round(statistics.fmean(char_lens), 2),
        "emoji_pct": round(_pct(has_emoji, n), 2),
        "emoji_mean": round(statistics.fmean(emoji_counts), 4),
        "hashtag_pct": round(_pct(has_hash, n), 2),
        "cue_hashtag_pct": round(_pct(has_cue, n), 2),
        "user_placeholder_pct": round(_pct(user_ph, n), 2),
        "emoji_in_literal": emoji_by_label[0],
        "emoji_in_sarcastic": emoji_by_label[1],
        "cue_in_literal": cue_by_label[0],
        "cue_in_sarcastic": cue_by_label[1],
    }


def token_length_histogram(split, width: int = 5) -> list:
    buckets: Counter = Counter()
    for text in split.texts:
        n = len(tokenize_tweet(text))
        buckets[(n // width) * width] += 1
    return sorted(buckets.items())


def render_text(rows: list[dict], hists: dict) -> str:
    lines = [
        "Twitter sarcasm splits",
        "======================",
        "",
        "Counts come from dataset/{train,test,subtest}_{sentence,label}.csv.",
        "Tokenization uses examples.common.tokenize (comma-to-space, then regex).",
        "",
    ]
    header = (
        f"{'split':8} {'n':>7} {'lit':>7} {'sarc':>7} "
        f"{'tok μ':>7} {'tok max':>7} {'emoji%':>8} {'#tag%':>7} {'cue%':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row['name']:8} {row['n']:7d} {row['n_literal']:7d} {row['n_sarcastic']:7d} "
            f"{row['tokens_mean']:7.2f} {row['tokens_max']:7d} "
            f"{row['emoji_pct']:8.2f} {row['hashtag_pct']:7.2f} {row['cue_hashtag_pct']:7.2f}"
        )
    lines.append("")
    lines.append("Emoji-containing tweets by label (count, not percent of class):")
    for row in rows:
        lines.append(
            f"  {row['name']}: literal={row['emoji_in_literal']}  "
            f"sarcastic={row['emoji_in_sarcastic']}"
        )
    lines.append("")
    lines.append("Cue-hashtag tweets by label:")
    for row in rows:
        lines.append(
            f"  {row['name']}: literal={row['cue_in_literal']}  "
            f"sarcastic={row['cue_in_sarcastic']}"
        )
    lines.append("")
    lines.append("Token-length histogram (train, bucket width 5):")
    for start, count in hists["train"]:
        bar = "#" * max(1, count // 400)
        lines.append(f"  {start:3d}-{start+4:<3d} {count:6d} {bar}")
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("* train is mildly imbalanced toward the literal class.")
    lines.append("* test is balanced 1000 / 1000.")
    lines.append("* subtest is the emoji-only slice: every row contains at least one pictograph.")
    lines.append("* cue hashtags (#sarcasm, #not, ...) are much denser in test than train.")
    return "\n".join(lines) + "\n"


def render_markdown(rows: list[dict], hists: dict) -> str:
    lines = [
        "# Dataset report (generated)",
        "",
        "Produced by `examples/01_explore_dataset.py`. Re-run that script after",
        "changing the CSV files.",
        "",
        "| split | n | literal | sarcastic | mean tokens | max tokens | emoji % | hashtag % | cue-hashtag % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['n']} | {row['n_literal']} | {row['n_sarcastic']} | "
            f"{row['tokens_mean']:.2f} | {row['tokens_max']} | {row['emoji_pct']:.2f} | "
            f"{row['hashtag_pct']:.2f} | {row['cue_hashtag_pct']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Why subtest exists",
            "",
            "The 2023 write-up reports a second test set of 278 tweets in which",
            "**every** example contains at least one emoji. That slice is the only",
            "place a word-only model and a word+emoji model can disagree because of",
            "the emoji channel. On the full test set most rows have no pictograph,",
            "so concatenating a 200-d zero vector cannot help.",
            "",
            "## Cue hashtags",
            "",
            "Cue tags (`#sarcasm`, `#not`, `#sarcastictweet`) are a known leakage",
            "source in Twitter sarcasm detection. They are present in both classes",
            "in this export (people also write `#not` in non-sarcastic complaints),",
            "but they are far more common on the sarcastic side of the test split.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="emit GitHub-flavored markdown")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    splits = list(iter_splits())
    rows = [summarize_split(s) for s in splits]
    hists = {s.name: token_length_histogram(s) for s in splits}
    if args.json:
        print(json.dumps({"splits": rows, "histograms": {k: hists[k] for k in hists}}, indent=2))
    elif args.markdown:
        print(render_markdown(rows, hists), end="")
    else:
        print(render_text(rows, hists), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
