#!/usr/bin/env python3
"""Print split sizes, class balance, length, and cue rates.

This is the first walkthrough: confirm the CSVs the notebooks trained
on, without GloVe or TensorFlow.

    python3 examples/inspect_dataset.py
    python3 examples/inspect_dataset.py --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset import load_all  # noqa: E402
from examples.lib.tokenize import tokenize_tweet  # noqa: E402

SARC_TAGS = {"#not", "#sarcasm", "#sarcastic", "#sarcastictweet"}


def has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def cue_counts(sentences: List[str]) -> Dict[str, int]:
    n_hash = n_user = n_sarc = n_nonascii = 0
    for sentence in sentences:
        tokens = tokenize_tweet(sentence)
        tags = {tok for tok in tokens if tok.startswith("#")}
        n_hash += int(bool(tags))
        n_user += int("<user>" in tokens)
        n_sarc += int(bool(tags & SARC_TAGS))
        n_nonascii += int(has_non_ascii(sentence))
    return {
        "with_hashtag": n_hash,
        "with_user": n_user,
        "with_sarc_hashtag": n_sarc,
        "with_non_ascii": n_nonascii,
    }


def tag_label_crosstab(sentences: List[str], labels: List[int]) -> Dict[str, int]:
    tagged_pos = tagged_neg = 0
    for sentence, label in zip(sentences, labels):
        tags = {tok for tok in tokenize_tweet(sentence) if tok.startswith("#")}
        if tags & SARC_TAGS:
            if label == 1:
                tagged_pos += 1
            else:
                tagged_neg += 1
    return {"tag_and_label1": tagged_pos, "tag_and_label0": tagged_neg}


def summarize() -> Dict[str, dict]:
    report = {}
    for name, split in load_all().items():
        labels = Counter(split.labels)
        ws_lens = [len(s.split()) for s in split.sentences]
        tok_lens = [len(tokenize_tweet(s)) for s in split.sentences]
        report[name] = {
            "n": split.n,
            "label_0": labels.get(0, 0),
            "label_1": labels.get(1, 0),
            "positive_rate": labels.get(1, 0) / split.n,
            "whitespace_tokens": {
                "mean": statistics.mean(ws_lens),
                "median": statistics.median(ws_lens),
                "max": max(ws_lens),
            },
            "walkthrough_tokens": {
                "mean": statistics.mean(tok_lens),
                "median": statistics.median(tok_lens),
                "max": max(tok_lens),
            },
            "cues": cue_counts(split.sentences),
            "sarc_hashtag_vs_label": tag_label_crosstab(split.sentences, split.labels),
        }
    return report


def _fmt_block(name: str, block: dict) -> str:
    ws = block["whitespace_tokens"]
    tk = block["walkthrough_tokens"]
    cues = block["cues"]
    xt = block["sarc_hashtag_vs_label"]
    lines = [
        f"=== {name} ===",
        f"n                 {block['n']}",
        f"label 0 / 1       {block['label_0']} / {block['label_1']}",
        f"positive rate     {block['positive_rate']:.3f}",
        (
            "whitespace len    "
            f"mean {ws['mean']:.2f}  median {ws['median']:.1f}  max {ws['max']}"
        ),
        (
            "walkthrough len   "
            f"mean {tk['mean']:.2f}  median {tk['median']:.1f}  max {tk['max']}"
        ),
        (
            "cues              "
            f"hashtag={cues['with_hashtag']}  <user>={cues['with_user']}  "
            f"sarc_tag={cues['with_sarc_hashtag']}  non_ascii={cues['with_non_ascii']}"
        ),
        (
            "sarc tag × label  "
            f"tag∧1={xt['tag_and_label1']}  tag∧0={xt['tag_and_label0']}"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the raw summary dict as JSON",
    )
    args = parser.parse_args()
    report = summarize()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("Twitter sarcasm CSVs (personal 2023 project)\n")
    for name in ("train", "test", "subtest"):
        print(_fmt_block(name, report[name]))
    print(
        "Walkthrough token counts use examples/lib/tokenize.py, not NLTK.\n"
        "See docs/dataset.md for the interpretation of each split."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
