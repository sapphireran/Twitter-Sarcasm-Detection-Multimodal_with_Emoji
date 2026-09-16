"""Fit a NumPy logistic model on surface features from the real CSVs.

This is a personal walkthrough, not a 2023 result. It exists so you can
get accuracy / F1 on the official test and emoji subtest without GloVe.
Cue hashtags are strong on purpose: the script can drop them so you can
see how much of the signal is just `#not` / `#sarcasm`.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .dataset_io import load_split
from .lexical import (
    FEATURE_NAMES,
    extract_feature_matrix,
    standardize,
    top_features_by_abs_weight,
)
from .logreg import binary_scores, fit_logreg, format_scores, predict_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train", type=int, default=0, help="0 = use every train row")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--drop-explicit-cues",
        action="store_true",
        help="zero #not / sarcasm / yeahright features before fitting",
    )
    return parser


_CUE_COLUMNS = {
    "has_not_hashtag",
    "has_sarcasm_hashtag",
    "has_yeahright",
}


def _maybe_drop_cues(matrix: np.ndarray) -> np.ndarray:
    keep = [name not in _CUE_COLUMNS for name in FEATURE_NAMES]
    return matrix[:, np.array(keep, dtype=bool)]


def _feature_names(drop_cues: bool) -> tuple:
    if not drop_cues:
        return FEATURE_NAMES
    return tuple(name for name in FEATURE_NAMES if name not in _CUE_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train = load_split("train")
    test = load_split("test")
    subtest = load_split("subtest")

    rng = np.random.default_rng(args.seed)
    train_idx = np.arange(train.n)
    if args.max_train and args.max_train < train.n:
        train_idx = rng.choice(train_idx, size=args.max_train, replace=False)
    else:
        train_idx = rng.permutation(train_idx)

    print(f"extracting lexical features ({len(FEATURE_NAMES)} dims)...")
    x_train = extract_feature_matrix([train.sentences[i] for i in train_idx])
    y_train = np.array([train.labels[i] for i in train_idx], dtype=int)
    x_test = extract_feature_matrix(test.sentences)
    y_test = np.array(test.labels, dtype=int)
    x_sub = extract_feature_matrix(subtest.sentences)
    y_sub = np.array(subtest.labels, dtype=int)

    if args.drop_explicit_cues:
        x_train = _maybe_drop_cues(x_train)
        x_test = _maybe_drop_cues(x_test)
        x_sub = _maybe_drop_cues(x_sub)
        names = _feature_names(True)
        print("dropped explicit cue columns:", ", ".join(sorted(_CUE_COLUMNS)))
    else:
        names = FEATURE_NAMES

    x_train_s, x_test_s, x_sub_s = standardize(x_train, x_test, x_sub)
    fit = fit_logreg(
        x_train_s,
        y_train,
        l2=args.l2,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(
        f"trained on {len(y_train)} tweets  "
        f"epochs={fit.n_epochs}  loss={fit.train_loss:.4f}"
    )
    print()

    train_pred = predict_label(x_train_s, fit.weights, fit.bias)
    test_pred = predict_label(x_test_s, fit.weights, fit.bias)
    sub_pred = predict_label(x_sub_s, fit.weights, fit.bias)
    print(format_scores("train  ", binary_scores(y_train, train_pred)))
    print(format_scores("test   ", binary_scores(y_test, test_pred)))
    print(format_scores("subtest", binary_scores(y_sub, sub_pred)))
    print()
    print("largest |weights| (standardized features)")
    ranked = top_features_by_abs_weight(
        fit.weights, k=min(8, len(names)), names=names
    )
    for name, weight in ranked:
        print(f"  {weight:+7.3f}  {name}")
    print()
    print(
        "These numbers are a lexical walkthrough. They are not the GloVe / "
        "Bi-LSTM results in docs/experiments.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
