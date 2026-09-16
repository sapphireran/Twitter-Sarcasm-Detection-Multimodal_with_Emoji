#!/usr/bin/env python3
"""Print split sizes, label balance, emoji rates, and cue rates."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.io_utils import load_all_splits
from examples.lib.tweet_features import (
    EMOJI_RE,
    HASHTAG_RE,
    SARCASM_HASHTAGS,
    extract_emojis,
    tokenize_tweet,
)


def _rate(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100 * count / total:5.1f}%"


def _emoji_histogram(texts: list[str], limit: int = 8) -> str:
    counts: collections.Counter[str] = collections.Counter()
    for text in texts:
        counts.update(extract_emojis(text))
    if not counts:
        return "(none)"
    return ", ".join(f"{glyph}×{n}" for glyph, n in counts.most_common(limit))


def report(markdown: bool = False) -> str:
    splits = load_all_splits()
    lines: list[str] = []

    def add(line: str = "") -> None:
        lines.append(line)

    add("# Dataset snapshot")
    add()
    add(
        "Computed live from `dataset/*.csv`. "
        "Label `1` is sarcastic; label `0` is not sarcastic."
    )
    add()

    if markdown:
        add("| Split | Tweets | Sarcastic | Non-sarcastic | Mean tokens | With emoji | With hashtag |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for name, split in splits.items():
        sarcastic = [text for text, label in split.pairs() if label == 1]
        sincere = [text for text, label in split.pairs() if label == 0]
        token_counts = [len(tokenize_tweet(text)) for text in split.texts]
        mean_tokens = sum(token_counts) / len(token_counts)
        emoji_n = sum(1 for text in split.texts if EMOJI_RE.search(text))
        hashtag_n = sum(1 for text in split.texts if HASHTAG_RE.search(text))
        if markdown:
            add(
                f"| {name} | {len(split)} | {len(sarcastic)} | {len(sincere)} | "
                f"{mean_tokens:.1f} | {_rate(emoji_n, len(split))} | {_rate(hashtag_n, len(split))} |"
            )
        else:
            add(f"## {name}")
            add(f"  tweets           {len(split)}")
            add(f"  sarcastic        {len(sarcastic)} ({_rate(len(sarcastic), len(split))})")
            add(f"  non-sarcastic    {len(sincere)} ({_rate(len(sincere), len(split))})")
            add(f"  mean tokens      {mean_tokens:.2f}")
            add(f"  with emoji       {emoji_n} ({_rate(emoji_n, len(split))})")
            add(f"  with hashtag     {hashtag_n} ({_rate(hashtag_n, len(split))})")
            add(f"  top emoji all    {_emoji_histogram(split.texts)}")
            add(f"  top emoji y=1    {_emoji_histogram(sarcastic)}")
            add(f"  top emoji y=0    {_emoji_histogram(sincere)}")
            add()

    add()
    add("## Cue rates by class")
    add()
    add("A cue is counted when the lowercase tweet contains the substring.")
    add()
    cues = ("#not", "#sarcasm", "#sarcastictweet", "love", "great")
    if markdown:
        add("| Split | Class | n | " + " | ".join(cues) + " |")
        add("| --- | ---: | ---: | " + " | ".join(["---:"] * len(cues)) + " |")

    for name, split in splits.items():
        for label, title in ((1, "sarcastic"), (0, "non-sarcastic")):
            rows = [text.lower() for text, y in split.pairs() if y == label]
            rates = [_rate(sum(cue in text for text in rows), len(rows)) for cue in cues]
            if markdown:
                add(f"| {name} | {title} | {len(rows)} | " + " | ".join(rates) + " |")
            else:
                pretty = ", ".join(f"{cue} {rate}" for cue, rate in zip(cues, rates, strict=True))
                add(f"  {name:7} {title:14} n={len(rows):5}  {pretty}")

    add()
    add("## Split overlap")
    add()
    train, test, subtest = splits["train"], splits["test"], splits["subtest"]
    train_set, test_set, sub_set = set(train.texts), set(test.texts), set(subtest.texts)
    add(f"- unique train tweets: {len(train_set)}")
    add(f"- unique test tweets: {len(test_set)}")
    add(f"- subtest tweets also in test: {len(sub_set & test_set)} / {len(sub_set)}")
    add(f"- train/test exact string overlap: {len(train_set & test_set)}")
    add()
    add(
        "Subtest is the emoji-bearing slice of the official test set. "
        "That is why multi-modal gains show up more clearly there."
    )
    add()
    add("## Explicit sarcasm hashtags")
    add()
    for name, split in splits.items():
        tagged = 0
        for text in split.texts:
            tags = {match.group(0).lower() for match in HASHTAG_RE.finditer(text)}
            tagged += int(bool(tags & SARCASM_HASHTAGS))
        add(f"- {name}: {tagged} tweets contain {sorted(SARCASM_HASHTAGS)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="emit GitHub-flavored tables instead of a plain-text dump",
    )
    parser.add_argument("-o", "--output", type=Path, help="optional file to write")
    args = parser.parse_args()
    text = report(markdown=args.markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
