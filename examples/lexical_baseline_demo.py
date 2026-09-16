#!/usr/bin/env python3
"""Train a tiny logistic model on lexical cues and score the official splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.io_utils import load_split
from examples.lib.lexical_baseline import (
    binary_metrics,
    featurize,
    fit_logistic,
)
from examples.lib.tweet_features import extract_features


def _evaluate(model, texts: list[str], labels: list[int]):
    features = featurize(texts)
    preds = model.predict(features)
    return binary_metrics(np.asarray(labels), preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = load_split("train")
    test = load_split("test")
    subtest = load_split("subtest")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train))
    chosen = order[: min(args.train_limit, len(train))]
    train_texts = [train.texts[i] for i in chosen]
    train_labels = np.asarray([train.labels[i] for i in chosen], dtype=int)

    print("Lexical logistic baseline")
    print("=========================")
    print(f"training on {len(train_texts)} random train tweets, {args.epochs} epochs")
    print("features are hashtags, emoji polarity, contrast, elongation, punctuation")
    print()

    model = fit_logistic(
        featurize(train_texts),
        train_labels,
        epochs=args.epochs,
        seed=args.seed,
    )

    print("largest |weights|")
    for name, value in model.top_weights(8):
        print(f"  {value:+.3f}  {name}")
    print()

    for split in (train, test, subtest):
        if split.name == "train":
            metrics = _evaluate(model, train_texts, train_labels.tolist())
            title = f"train subset (n={len(train_texts)})"
        else:
            metrics = _evaluate(model, split.texts, split.labels)
            title = f"{split.name} (n={len(split)})"
        row = metrics.as_percent_row()
        print(
            f"{title:28} acc={row['accuracy']:>6}  "
            f"p={row['precision']:>6}  r={row['recall']:>6}  f1={row['f1']:>6}"
        )

    print()
    sample_path = Path(__file__).with_name("sample_tweets.json")
    samples = json.loads(sample_path.read_text(encoding="utf-8"))["tweets"]
    print("hand-picked examples")
    for tweet in samples:
        vector = np.asarray([extract_features(tweet["text"]).as_vector()])
        prob = float(model.predict_proba(vector)[0])
        pred = int(prob >= 0.5)
        mark = "ok" if pred == tweet["label"] else "miss"
        print(
            f"  [{mark}] p={prob:.2f} gold={tweet['label']} pred={pred}  "
            f"{tweet['id']}: {tweet['text'][:72]}"
        )
    print()
    print(
        "This baseline is weaker than Bi-LSTM + attention (87%+ test accuracy "
        "in the 2023 notebooks) but it makes the #not / contrast story visible."
    )


if __name__ == "__main__":
    main()
