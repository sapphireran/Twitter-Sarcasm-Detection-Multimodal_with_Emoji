#!/usr/bin/env python3
"""Hashtag and clash heuristics that run on the CSVs alone.

These are *not* the 2023 course models. They exist so you can see how much
of the official test/subtest signal is just distant-supervision tags.

Rules
-----
majority   always predict the training majority label (0)
cue-tag    sarcastic iff a known sarcasm hashtag is present
clash      sarcastic iff a cue-tag *or* (positive word + negative word/emoji)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    SPLITS,
    binary_scores,
    format_scores,
    load_split,
    majority_label,
    sarcasm_cue_tags,
    simple_tweet_tokens,
)

POSITIVE = {
    "love",
    "loovee",
    "loved",
    "loves",
    "great",
    "greatest",
    "best",
    "awesome",
    "yay",
    "yayy",
    "wonderful",
    "perfect",
    "glad",
    "happy",
    "excited",
    "can't",
    "cant",
    "wait",
    "nice",
    "fantastic",
    "amazing",
}

NEGATIVE = {
    "hate",
    "hated",
    "dirty",
    "late",
    "tired",
    "pissed",
    "useless",
    "worst",
    "sore",
    "disappointment",
    "disappointed",
    "exhausted",
    "broken",
    "sick",
    "stupid",
    "dumb",
    "ugly",
    "3am",
    "never",
    "not",
}

NEGATIVE_EMOJI = set("😒😭🔫💔😡😠😞🙁😟😩😫🙄😑")


def predict_majority(texts, majority: int):
    return [majority] * len(texts)


def predict_cue_tag(texts):
    return [1 if sarcasm_cue_tags(t) else 0 for t in texts]


def predict_clash(texts):
    out = []
    for text in texts:
        if sarcasm_cue_tags(text):
            out.append(1)
            continue
        tokens = set(simple_tweet_tokens(text))
        has_pos = bool(tokens & POSITIVE)
        has_neg = bool(tokens & NEGATIVE) or any(ch in NEGATIVE_EMOJI for ch in text)
        out.append(1 if (has_pos and has_neg) else 0)
    return out


RULES = {
    "majority": predict_majority,
    "cue-tag": predict_cue_tag,
    "clash": predict_clash,
}


def evaluate_split(split: str, majority: int) -> list[dict]:
    texts, labels = load_split(split)
    rows = []
    for name, fn in RULES.items():
        preds = fn(texts, majority) if name == "majority" else fn(texts)
        scores = binary_scores(labels, preds)
        scores["rule"] = name
        scores["split"] = split
        rows.append(scores)
    return rows


def show_errors(split: str, rule: str, n: int, majority: int) -> None:
    texts, labels = load_split(split)
    fn = RULES[rule]
    preds = fn(texts, majority) if rule == "majority" else fn(texts)
    print(f"\n=== {n} errors on {split} / {rule} ===")
    shown = 0
    for text, y, p in zip(texts, labels, preds):
        if int(y) == int(p):
            continue
        kind = "FP" if p == 1 else "FN"
        print(f"[{kind}] gold={int(y)} pred={int(p)}  {text}")
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("(none)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=SPLITS,
        action="append",
        help="Evaluate only these splits. Default: test and subtest, plus train as a sanity check.",
    )
    parser.add_argument(
        "--errors",
        type=int,
        default=0,
        help="Print this many cue-tag errors per selected split.",
    )
    parser.add_argument(
        "--error-rule",
        choices=tuple(RULES),
        default="cue-tag",
        help="Which rule to sample errors from.",
    )
    args = parser.parse_args(argv)

    train_texts, train_labels = load_split("train")
    majority = majority_label(train_labels)
    splits = args.split or ["train", "test", "subtest"]

    print(
        f"Training majority label = {majority} "
        f"(n={len(train_labels)}). "
        "cue-tag uses #not/#sarcasm/#yeahright and friends. "
        "clash adds a small positive-vs-negative lexicon.\n"
    )
    print("These numbers are live. They are not the Bi-LSTM / RF table.\n")

    for split in splits:
        print(f"=== {split} ===")
        for row in evaluate_split(split, majority):
            print("  " + format_scores(row["rule"], row))
        if args.errors:
            show_errors(split, args.error_rule, args.errors, majority)
        print()

    print(
        "Reading the table: cue-tag precision on test/subtest is 1.0 in this "
        "snapshot (every cue hashtag is on a positive label) while recall is "
        "only the tagged fraction of sarcastic tweets. A neural net that "
        "cannot beat cue-tag *precision* is not beating the annotation rule."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
