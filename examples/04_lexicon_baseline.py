#!/usr/bin/env python3
"""Hashtag lexicon baseline vs majority class.

Predict sarcastic if the tweet contains ``#not``, ``#sarcasm``,
``#sarcastictweet``, ``#yeahright``, or an irony hashtag. This is the
honest floor for the course dataset: the test/subtest sarcastic class is
full of those self-annotation tags.

The BiLSTM numbers in ``examples/reported_results.json`` are the ceiling
the 2023 project actually hit. The gap is the interesting part.
"""

from __future__ import annotations

import _path  # noqa: F401

from sarcasm_toolkit.baseline import LexiconBaseline
from sarcasm_toolkit.cues import CUE_HASHTAGS
from sarcasm_toolkit.dataset import load_split
from sarcasm_toolkit.metrics import binary_metrics, format_metrics, majority_baseline
from sarcasm_toolkit.reported import format_reported_table


def evaluate(name: str, model: LexiconBaseline) -> None:
    split = load_split(name)
    pred = model.predict(split.texts)
    majority = majority_baseline(split.labels)
    print(format_metrics(f"lexicon/{name}", binary_metrics(split.labels, pred)))
    print(format_metrics(f"majority/{name}", binary_metrics(split.labels, majority)))


def main() -> None:
    print("Lexicon hashtag baseline")
    print("========================")
    print("Cue tags:", ", ".join(sorted(CUE_HASHTAGS)))
    print()
    model = LexiconBaseline()
    for name in ("train", "test", "subtest"):
        evaluate(name, model)

    print("\nHow to read this")
    print("----------------")
    print("* High precision, lower recall on train: most #not tweets are")
    print("  sarcastic, but plenty of sarcastic tweets have no cue tag.")
    print("* Test/subtest recall jumps because sarcastic tweets there")
    print("  were collected with those hashtags (distant supervision).")
    print("* The 2023 BiLSTM+attention still beats this floor, especially")
    print("  on tweets that do not advertise the label in a hashtag.")
    print()
    print(format_reported_table())


if __name__ == "__main__":
    main()
