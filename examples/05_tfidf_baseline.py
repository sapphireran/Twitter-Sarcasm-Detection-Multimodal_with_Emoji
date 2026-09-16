#!/usr/bin/env python3
"""Laptop-scale hashed TF-IDF logistic baseline on the shipped CSVs.

This is not a reimplementation of the 2023 GloVe SVM. It exists so a clone
of the repo can produce an accuracy number without the Twitter GloVe dump.
Hashtag tokens are strong features; the script also prints an ablation that
drops them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.hashed_tfidf import (
    estimate_idf,
    evaluate,
    featurize_text,
    fit_logistic,
    hashed_counts,
    to_sparse,
)
from examples.lib.io import load_split
from examples.lib.tokenize import tokenize


def _prepare(
    rows: list[tuple[str, int]],
    hash_size: int,
    drop_hashtags: bool,
    idf: np.ndarray | None = None,
):
    texts = [text for text, _ in rows]
    labels = np.fromiter((label for _, label in rows), dtype=np.int64, count=len(rows))
    count_rows = []
    for text in texts:
        tokens = tokenize(text)
        if drop_hashtags:
            tokens = [token for token in tokens if not token.startswith("#")]
        count_rows.append(hashed_counts(tokens, hash_size))
    if idf is None:
        idf = estimate_idf(count_rows, hash_size)
    sparse = [to_sparse(counts, idf=idf) for counts in count_rows]
    return texts, labels, sparse, idf


def _run_setting(
    train_rows: list[tuple[str, int]],
    eval_splits: dict[str, list[tuple[str, int]]],
    hash_size: int,
    epochs: int,
    seed: int,
    drop_hashtags: bool,
) -> None:
    title = "hashtags dropped" if drop_hashtags else "all tokens"
    print(f"## {title}")
    texts_tr, y_tr, rows_tr, idf = _prepare(train_rows, hash_size, drop_hashtags)
    model = fit_logistic(
        rows_tr,
        y_tr,
        hash_size=hash_size,
        idf=idf,
        drop_hashtags=drop_hashtags,
        epochs=epochs,
        seed=seed,
    )
    train_metrics = evaluate(model, texts_tr, y_tr)
    print(f"  train   {train_metrics.as_row()}")
    for split_name, rows in eval_splits.items():
        texts = [text for text, _ in rows]
        labels = np.fromiter((label for _, label in rows), dtype=np.int64, count=len(rows))
        print(f"  {split_name:<7} {evaluate(model, texts, labels).as_row()}")
    # Touch featurize_text so the public helper stays in the hot path.
    _ = featurize_text(texts_tr[0], hash_size, idf=idf, drop_hashtags=drop_hashtags)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="If > 0, use only the first N training rows (debug).",
    )
    args = parser.parse_args()

    train = load_split("train")
    if args.max_train > 0:
        train = train[: args.max_train]
    eval_splits = {"test": load_split("test"), "subtest": load_split("subtest")}

    print(
        f"hashed TF-IDF logistic  hash_size={args.hash_size}  "
        f"epochs={args.epochs}  train_rows={len(train)}\n"
        "Not comparable to the GloVe SVM in docs/baselines-and-results.md.\n"
    )
    for drop in (False, True):
        _run_setting(
            train_rows=train,
            eval_splits=eval_splits,
            hash_size=args.hash_size,
            epochs=args.epochs,
            seed=args.seed,
            drop_hashtags=drop,
        )
    print("The drop from 'all tokens' to 'hashtags dropped' is the leak size.")


if __name__ == "__main__":
    main()
