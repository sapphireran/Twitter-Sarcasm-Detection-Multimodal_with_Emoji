#!/usr/bin/env python3
"""Measure the hashtag leak: how much of test sarcasm is named in-band.

On this corpus, every test tweet that contains #not / #sarcasm / …
is labeled sarcastic. A classifier that only looks at those tags is
therefore a serious baseline — and a confounder for any 'language
understanding' claim. This script prints the contingency tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (  # noqa: E402
    CUE_HASHTAGS,
    has_cue_hashtag,
    is_emoji_token,
    read_sentence_label_pair,
    repo_root,
)


def contingency(split: str) -> None:
    root = repo_root()
    docs, labels = read_sentence_label_pair(
        root / "dataset" / f"{split}_sentence.csv",
        root / "dataset" / f"{split}_label.csv",
    )
    tp = fp = tn = fn = 0
    hard = 0  # sarcastic, no cue, no emoji
    for doc, y in zip(docs, labels):
        pred = int(has_cue_hashtag(doc))
        y = int(y)
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif (not pred) and (not y):
            tn += 1
        else:
            fn += 1
        if y and not pred and not any(is_emoji_token(t) for t in doc):
            hard += 1
    n = len(labels)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / n
    print(f"=== {split}  n={n}  cue set={sorted(CUE_HASHTAGS)}")
    print(f"cue-as-predictor  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  sarcastic with neither cue nor emoji (harder slice): {hard}")
    print()


def main() -> int:
    print("Rule: predict sarcastic iff the tweet has a cue hashtag.")
    print("That rule is a leak check, not a proposed system.\n")
    for split in ("train", "test", "subtest"):
        contingency(split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
