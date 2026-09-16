#!/usr/bin/env python3
"""End-to-end toy version of the project's single- vs multi-modal setup.

Steps, matching ``ml_read_data`` then a linear classifier:

1. Read ``examples/fixtures/tiny_tweets.csv`` + labels.
2. Tokenize each tweet.
3. Mean-pool against a tiny GloVe-like table (single-modal, 8-d).
4. Mean-pool against a tiny emoji2vec-like table and concatenate
   (multi-modal, 16-d).
5. Fit logistic regression on a train slice; score the held-out slice.

The numbers are not the 2023 test-set scores. They exist so you can step
through the same *shape* of experiment without downloading GloVe.

Run from the repository root:

    python examples/toy_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.classify import accuracy, f1_score, fit_logreg, predict_label
from examples.dataset_io import read_labels, read_sentence_strings
from examples.embeddings import average_corpus, multimodal_concat
from examples.fixtures.tiny_tables import DIM, EMOJI_TABLE, WORD_TABLE
from examples.tokenize import tokenize_tweet

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def load_docs() -> tuple[list[list[str]], np.ndarray]:
    texts = read_sentence_strings(FIXTURE_DIR / "tiny_tweets.csv")
    labels = np.asarray(read_labels(FIXTURE_DIR / "tiny_labels.csv"), dtype=int)
    docs = [tokenize_tweet(text) for text in texts]
    return docs, labels


def features_for(docs: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    word = average_corpus(docs, WORD_TABLE, dim=DIM)
    emoji = average_corpus(docs, EMOJI_TABLE, dim=DIM)
    return word, multimodal_concat(word, emoji)


def stratified_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """6 sarcastic + 6 non-sarcastic for train; 2 + 2 for test.

    ``ml_read_data`` shuffled the full corpus. With only 16 fixture rows
    a prefix split would put every sarcastic tweet in train and make F1
    undefined on test, so the demo stratifies instead.
    """
    sarcastic = np.flatnonzero(labels == 1)
    non = np.flatnonzero(labels == 0)
    if len(sarcastic) < 8 or len(non) < 8:
        raise ValueError("fixture labels must contain 8 of each class")
    train = np.concatenate([sarcastic[:6], non[:6]])
    test = np.concatenate([sarcastic[6:], non[6:]])
    return train, test


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name:<18} acc={acc:.3f}  f1={f1:.3f}  pred={y_pred.tolist()}")


def main() -> int:
    docs, labels = load_docs()
    word, both = features_for(docs)
    train, test = stratified_split(labels)

    print(f"docs={len(docs)}  word_dim={word.shape[1]}  multimodal_dim={both.shape[1]}")
    print(f"train idx {train.tolist()}  labels {labels[train].tolist()}")
    print(f"test  idx {test.tolist()}  labels {labels[test].tolist()}")
    print()

    w_word, b_word = fit_logreg(word[train], labels[train])
    w_both, b_both = fit_logreg(both[train], labels[train])

    pred_word = predict_label(word[test], w_word, b_word)
    pred_both = predict_label(both[test], w_both, b_both)

    report("single-modal", labels[test], pred_word)
    report("multi-modal", labels[test], pred_both)
    print()
    print("Largest |weight| axes in the multi-modal model (16-d = word|emoji):")
    order = np.argsort(-np.abs(w_both))
    axis_names = [
        "w:pos",
        "w:neg",
        "w:sarc-tag",
        "w:grind",
        "w:affection",
        "w:sad-emoji",
        "w:deadpan-emoji",
        "w:happy-emoji",
        "e:pos",
        "e:neg",
        "e:sarc-tag",
        "e:grind",
        "e:affection",
        "e:sad-emoji",
        "e:deadpan-emoji",
        "e:happy-emoji",
    ]
    for index in order[:6]:
        print(f"  {axis_names[index]:<16} {w_both[index]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
