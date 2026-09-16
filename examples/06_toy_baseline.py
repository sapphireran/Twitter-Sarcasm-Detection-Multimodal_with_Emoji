#!/usr/bin/env python3
"""Nearest-centroid baseline on cue / emoji / length features.

This is the cheap leak check described in docs/experiments.md. It is
trained on the official train split and scored on test / subtest, but
the *features* are only:

    [has_cue_hashtag, has_emoji, token_count, elongated_love]

No GloVe, no emoji2vec, no sklearn. A high test score here means the
label is sitting in the hashtag, not that a 2.5M-param BiLSTM is doing
deep pragmatic inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (  # noqa: E402
    accuracy,
    cue_emoji_features,
    nearest_centroid_predict,
    read_sentence_label_pair,
    repo_root,
)


def load(split: str):
    root = repo_root()
    docs, labels = read_sentence_label_pair(
        root / "dataset" / f"{split}_sentence.csv",
        root / "dataset" / f"{split}_label.csv",
    )
    return cue_emoji_features(docs), labels, docs


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def main() -> int:
    x_tr, y_tr, _ = load("train")
    print(f"train features {x_tr.shape}  mean={x_tr.mean(axis=0).round(3)}")
    print("feature order: cue, emoji, length, lo+ve+")
    print()
    print(f"{'split':<8} {'acc':>7} {'f1':>7} {'pred_pos':>9} {'true_pos':>9}")
    for split in ("train", "test", "subtest"):
        x, y, _ = load(split)
        pred = nearest_centroid_predict(x_tr, y_tr, x)
        print(
            f"{split:<8} {accuracy(y, pred):7.3f} {f1(y, pred):7.3f} "
            f"{int(pred.sum()):9d} {int(y.sum()):9d}"
        )

    # Also the pure rule: cue hashtag ⇒ sarcastic (feature 0).
    print()
    print("rule cue⇒sarcastic")
    for split in ("train", "test", "subtest"):
        x, y, _ = load(split)
        pred = (x[:, 0] > 0.5).astype(np.int64)
        print(f"  {split:<8} acc={accuracy(y, pred):.3f}  f1={f1(y, pred):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
