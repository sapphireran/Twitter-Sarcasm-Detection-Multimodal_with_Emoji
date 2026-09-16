#!/usr/bin/env python3
"""Bag-of-tokens Naive Bayes on the shipped CSVs (no GloVe, no TF).

This is *not* the 2023 system. It is a cheap control that answers:

1. How far do unigram counts get you on this export?
2. How much of that accuracy disappears when cue hashtags are stripped?
3. Does adding a binary "has-emoji" feature move the needle on subtest?

The original SVM / RF / GBT baselines average 200-d GloVe (and optionally
200-d emoji2vec) into a tweet vector. Those pickle files are in
``baseline_models/`` but need Gensim + the GloVe binary to rebuild features.
This script only needs numpy and the CSVs.

Run::

    python3 examples/06_bow_baseline.py
    python3 examples/06_bow_baseline.py --max-features 2500 --min-count 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.bow import CountVectorizer, MultinomialNB, accuracy, f1_binary
from examples.common.emoji import emoji_code_points
from examples.common.io import load_split
from examples.common.tokenize import strip_hashtags, tokenize_tweet


def docs_for(split, mode: str) -> list:
    docs = []
    for text in split.texts:
        if mode == "stripped":
            text = strip_hashtags(text)
        tokens = tokenize_tweet(text)
        if mode == "emoji_flag":
            tokens = list(tokens)
            if emoji_code_points(text):
                tokens.append("__HAS_EMOJI__")
            tokens.extend(f"__E:{e}__" for e in emoji_code_points(text))
        docs.append(tokens)
    return docs


def eval_mode(mode: str, max_features: int, min_count: int) -> list[dict]:
    train = load_split("train")
    test = load_split("test")
    sub = load_split("subtest")
    train_docs = docs_for(train, mode)
    vec = CountVectorizer.fit(train_docs, min_count=min_count, max_features=max_features)
    x_train = vec.transform(train_docs)
    y_train = np.asarray(train.labels)
    clf = MultinomialNB.fit(x_train, y_train)
    rows = []
    for split, docs in (
        (test, docs_for(test, mode)),
        (sub, docs_for(sub, mode)),
    ):
        x = vec.transform(docs)
        pred = clf.predict(x)
        y = np.asarray(split.labels)
        rows.append(
            {
                "mode": mode,
                "split": split.name,
                "vocab": len(vec.vocab),
                "acc": accuracy(y, pred),
                "f1": f1_binary(y, pred),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()

    modes = [
        ("full", "keep hashtags, ignore emoji as a typed feature"),
        ("stripped", "strip #hashtags before tokenizing (cue ablation)"),
        ("emoji_flag", "keep hashtags + add __HAS_EMOJI__ and __E:…__ features"),
    ]
    print(
        f"{'mode':12} {'split':8} {'vocab':>6} {'acc':>8} {'F1':>8}  note"
    )
    print("-" * 78)
    for mode, note in modes:
        for row in eval_mode(mode, args.max_features, args.min_count):
            print(
                f"{row['mode']:12} {row['split']:8} {row['vocab']:6d} "
                f"{100*row['acc']:8.2f} {100*row['f1']:8.2f}  {note if row['split']=='test' else ''}"
            )
    print()
    print("How to read this")
    print("-----------------")
    print("* `full` is an upper bound on what a linear bag-of-words model can do")
    print("  while still seeing #sarcasm / #not. Compare to the SVM word-only")
    print("  accuracy of 76.9% on test (docs/04-results.md).")
    print("* `stripped` is the honest lexical baseline. If it is close to chance,")
    print("  the original labels are mostly distant-supervision hashtags.")
    print("* `emoji_flag` is a crude stand-in for the multimodal concatenation")
    print("  used by the sklearn baselines (200-d GloVe avg || 200-d emoji2vec avg).")
    print("  A real gain should show up on subtest, not just on test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
