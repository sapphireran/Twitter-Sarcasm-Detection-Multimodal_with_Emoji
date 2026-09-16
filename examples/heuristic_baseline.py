#!/usr/bin/env python3
"""Train a Bernoulli NB heuristic on hand-built features.

Two models are fit on the training split:

1. **with leak hashtags** — ``#not`` / ``#sarcasm`` / … are features
2. **without leak hashtags** — those features are forced off

The gap between the two on the same test set is the distant-supervision leakage
the 2023 notebooks never measured. This is a didactic baseline, not a claim
that Naive Bayes beats the BiLSTM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.features import feature_matrix  # noqa: E402
from lib.io import load_split  # noqa: E402
from lib.metrics import accuracy, f1, majority_baseline, precision, recall  # noqa: E402
from lib.naive_bayes import BernoulliNB  # noqa: E402


def _evaluate(name, model, texts, labels) -> None:
    include_leak = "no-leak" not in name
    pred = model.predict(feature_matrix(texts, include_leak=include_leak))
    print(
        f"  {name:<28} acc={accuracy(labels, pred):.3f}  "
        f"f1={f1(labels, pred):.3f}  "
        f"p={precision(labels, pred):.3f}  "
        f"r={recall(labels, pred):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    args = parser.parse_args()

    train = load_split("train")
    test = load_split("test")
    subtest = load_split("subtest")

    with_leak = BernoulliNB(alpha=args.alpha).fit(
        feature_matrix(train.texts, include_leak=True), train.labels
    )
    no_leak = BernoulliNB(alpha=args.alpha).fit(
        feature_matrix(train.texts, include_leak=False), train.labels
    )

    print("Top log-odds features (sarcastic class, leak model)")
    for name, score in with_leak.debug_top_features(8):
        print(f"  {score:+.3f}  {name}")

    print("\nMajority-class floors")
    print(f"  test    {majority_baseline(test.labels):.3f}")
    print(f"  subtest {majority_baseline(subtest.labels):.3f}")

    print("\nTest (2,000 balanced tweets)")
    _evaluate("bernoulli + leak tags", with_leak, test.texts, test.labels)
    _evaluate("bernoulli no-leak tags", no_leak, test.texts, test.labels)

    print("\nSubtest (278 emoji-heavy tweets)")
    _evaluate("bernoulli + leak tags", with_leak, subtest.texts, subtest.labels)
    _evaluate("bernoulli no-leak tags", no_leak, subtest.texts, subtest.labels)

    print(
        "\nReading the gap: if the leak model is much better, the original "
        "notebooks were partly detecting sarcasm hashtags rather than sarcasm."
    )


if __name__ == "__main__":
    main()
