#!/usr/bin/env python3
"""Train a NumPy logistic regression on surface cues only.

This is a *floor* for the coursework models: it uses the hashtags,
emoji counts, and elongation patterns documented in
docs/annotated_examples.md, and never touches GloVe or emoji2vec.

Usage:

    python3 examples/lexical_sarcasm_baseline.py
    python3 examples/lexical_sarcasm_baseline.py --seed 7 --epochs 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset_io import load_split  # noqa: E402
from examples.lib.lexical_features import (  # noqa: E402
    FEATURE_NAMES,
    hashtag_rule_predict,
    lexical_feature_matrix,
)
from examples.lib.logreg import LogisticRegressionGD  # noqa: E402


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _fmt(m: dict[str, float]) -> str:
    return (
        f"acc={m['acc']:.3f}  f1={m['f1']:.3f}  "
        f"prec={m['precision']:.3f}  rec={m['recall']:.3f}"
    )


def _top_weights(weights: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
    order = np.argsort(np.abs(weights))[::-1]
    return [(FEATURE_NAMES[i], float(weights[i])) for i in order[:k]]


def evaluate_split(
    name: str,
    model: LogisticRegressionGD,
    sentences: list[str],
    labels: list[int],
) -> None:
    x = lexical_feature_matrix(sentences)
    y = np.asarray(labels)
    pred = model.predict(x)
    rule = np.array([hashtag_rule_predict(s) for s in sentences])
    print(f"{name:8}  logreg  {_fmt(_metrics(y, pred))}")
    print(f"{'':8}  hashtag {_fmt(_metrics(y, rule))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--limit-train", type=int, default=0, help="0 = all rows")
    args = parser.parse_args(argv)

    train = load_split("train", args.dataset_dir)
    test = load_split("test", args.dataset_dir)
    sub = load_split("subtest", args.dataset_dir)

    sentences = train.sentences
    labels = train.labels
    if args.limit_train:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(sentences), size=args.limit_train, replace=False)
        sentences = [sentences[i] for i in idx]
        labels = [labels[i] for i in idx]

    x_train = lexical_feature_matrix(sentences)
    y_train = np.asarray(labels)
    model = LogisticRegressionGD(
        lr=args.lr, epochs=args.epochs, seed=args.seed
    ).fit(x_train, y_train)

    print(
        f"trained on {len(sentences):,} tweets, "
        f"{x_train.shape[1]} features, "
        f"final loss {model.loss_history[-1]:.4f}"
    )
    print("largest |weights|:")
    for name, value in _top_weights(model.weights):
        print(f"  {value:+.3f}  {name}")
    print()
    evaluate_split("train", model, sentences, labels)
    evaluate_split("test", model, test.sentences, test.labels)
    evaluate_split("subtest", model, sub.sentences, sub.labels)
    print()
    print(
        "The hashtag rule is `#not` / `#sarcasm*` / `#yeahright` → sarcastic. "
        "On these CSVs the rule has precision 1.0 on test and beats the "
        "train-fit logreg there — test is much more tag-saturated than "
        "train. See docs/results.md and docs/dataset.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
