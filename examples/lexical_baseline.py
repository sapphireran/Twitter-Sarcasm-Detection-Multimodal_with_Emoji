#!/usr/bin/env python3
"""Train bag-of-words baselines on the committed splits (no GloVe)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sarcasm_lab.cue_model import CountCueClassifier, RuleCueClassifier
from sarcasm_lab.cues import extract_cues
from sarcasm_lab.io import load_all_splits
from sarcasm_lab.metrics import binary_metrics, format_metrics
from sarcasm_lab.naive_bayes import MultinomialNB
from sarcasm_lab.sgd_logreg import SGDLogisticRegression
from sarcasm_lab.tables import format_table
from sarcasm_lab.vectorize import CountVectorizer


def _predict_rows(name: str, y_true: list[int], y_pred: list[int]) -> list[str]:
    m = binary_metrics(y_true, y_pred)
    row = m.as_percent_row()
    return [name, row["n"], row["acc"], row["p"], row["r"], row["f1"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-features", type=int, default=15000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--bigrams", action="store_true", help="add adjacent token pairs")
    args = parser.parse_args()

    splits = load_all_splits(tokenize=True)
    train, test, subtest = splits["train"], splits["test"], splits["subtest"]

    vectorizer = CountVectorizer(
        min_df=args.min_df,
        max_features=args.max_features,
        add_bigrams=args.bigrams,
    )
    X_train = vectorizer.fit_transform(train.tokens)
    X_test = vectorizer.transform(test.tokens)
    X_sub = vectorizer.transform(subtest.tokens)
    print(f"vocabulary: {vectorizer.n_features} tokens (min_df={args.min_df})")

    nb = MultinomialNB(alpha=1.0).fit(X_train, train.labels, vectorizer.n_features)
    logreg = SGDLogisticRegression(lr=0.05, l2=1e-4, epochs=args.epochs, seed=0)
    logreg.fit(X_train, train.labels, vectorizer.n_features)

    train_cues = [extract_cues(t, tok) for t, tok in zip(train.texts, train.tokens)]
    test_cues = [extract_cues(t, tok) for t, tok in zip(test.texts, test.tokens)]
    sub_cues = [extract_cues(t, tok) for t, tok in zip(subtest.texts, subtest.tokens)]
    rule = RuleCueClassifier()
    counted = CountCueClassifier()

    rows = [
        _predict_rows("NB / test", test.labels, nb.predict(X_test)),
        _predict_rows("NB / subtest", subtest.labels, nb.predict(X_sub)),
        _predict_rows("SGD-logreg / test", test.labels, logreg.predict(X_test)),
        _predict_rows("SGD-logreg / subtest", subtest.labels, logreg.predict(X_sub)),
        _predict_rows("rule-cues / test", test.labels, rule.predict(test_cues)),
        _predict_rows("rule-cues / subtest", subtest.labels, rule.predict(sub_cues)),
        _predict_rows("count-cues / test", test.labels, counted.predict(test_cues)),
        _predict_rows("count-cues / subtest", subtest.labels, counted.predict(sub_cues)),
    ]
    print()
    print(
        format_table(
            ["model", "n", "acc%", "p%", "r%", "f1%"],
            rows,
            right_align={1, 2, 3, 4, 5},
        )
    )
    print()
    print(format_metrics("NB train (overfit check)", binary_metrics(train.labels, nb.predict(X_train))))
    print(
        format_metrics(
            "rule-cues train",
            binary_metrics(train.labels, rule.predict(train_cues)),
        )
    )
    print()
    print("These numbers are lexical / cue baselines, not the 2023 GloVe models.")
    print("See docs/results.md for the Bi-LSTM + attention figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
