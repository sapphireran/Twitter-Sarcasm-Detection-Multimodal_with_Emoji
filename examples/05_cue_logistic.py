#!/usr/bin/env python3
"""Train a from-scratch logistic regressor on lexical sarcasm cues.

Features are the 17 flags in ``sarcasm_toolkit.cues`` (cue hashtags,
emoji counts, elongation, ``love when`` frames, …). Training is batch
gradient descent in pure Python so this example does not need sklearn.

Default: fit on 4k shuffled training rows (seed=2023) so the script
finishes in a few seconds. Pass ``--full`` to use all 39,780 rows.
"""

from __future__ import annotations

import argparse
import random

import _path  # noqa: F401

from sarcasm_toolkit.baseline import CueLogistic, LexiconBaseline, describe_prediction
from sarcasm_toolkit.dataset import load_split
from sarcasm_toolkit.metrics import binary_metrics, format_metrics

SAMPLE_TEXTS = [
    "I just love having grungy ass hair 😑 #not",
    "Being sore is the best and the worst feeling in the world",
    "<user> i hope youre lurking rn pretty please?! 😭 😭 😭",
    "Happy birthday to me. Yay.",
]


def subset(texts, labels, n: int, seed: int):
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    picked = order[:n]
    return [texts[i] for i in picked], [labels[i] for i in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="train on all train rows")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    train = load_split("train")
    if args.full:
        x_train, y_train = list(train.texts), list(train.labels)
    else:
        x_train, y_train = subset(train.texts, train.labels, args.n_train, seed=2023)

    print("Cue logistic (stdlib gradient descent)")
    print(f"train rows used: {len(x_train)}  epochs={args.epochs}")
    model = CueLogistic(epochs=args.epochs, learning_rate=0.3, l2=0.01)
    model.fit(x_train, y_train)

    print("\nLargest |weights| after standardization:")
    for name, weight in model.top_weights(10):
        print(f"  {weight:+.3f}  {name}")

    lexicon = LexiconBaseline()
    print("\nHeld-out splits")
    for name in ("test", "subtest"):
        split = load_split(name)
        print(format_metrics(f"logistic/{name}", binary_metrics(split.labels, model.predict(split.texts))))
        print(format_metrics(f"lexicon/{name} ", binary_metrics(split.labels, lexicon.predict(split.texts))))

    print("\nWalkthrough predictions")
    for text in SAMPLE_TEXTS:
        info = describe_prediction(text, model)
        print(f"\n  pred={info['prediction']} p={info['probability']:.3f}  cues={info['cue_hashtags']}")
        print(f"  {text}")


if __name__ == "__main__":
    main()
