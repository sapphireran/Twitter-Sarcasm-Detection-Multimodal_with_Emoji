#!/usr/bin/env python3
"""Hashed logistic baseline with optional cue features and slice metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccs2lab.hashed_logreg import fit_logreg
from ccs2lab.metrics import binary_scores
from ccs2lab.report import markdown_table
from ccs2lab.sample import stratified_take
from ccs2lab.slices import slice_scores
from ccs2lab.splits import load_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="0 = all train rows. Uses a stratified shuffle; a raw prefix is almost all sincere.",
    )
    parser.add_argument("--hash-dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-cues", action="store_true")
    args = parser.parse_args()

    bundle = load_bundle()
    train_texts, train_labels = stratified_take(
        bundle.train.texts,
        bundle.train.labels,
        args.train_limit,
        seed=args.seed,
    )

    model = fit_logreg(
        train_texts,
        train_labels,
        hash_dim=args.hash_dim,
        use_cues=not args.no_cues,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(
        f"fitted hashed logreg  n_train={len(train_texts)}  "
        f"hash_dim={args.hash_dim}  cues={not args.no_cues}  "
        f"epochs={args.epochs}  seed={args.seed}"
    )
    if model.use_cues:
        print("cue weights (abs-sorted):")
        for name, weight in model.top_cue_weights():
            print(f"  {name:16s} {weight:+.3f}")
    print()

    headers = ("split", "slice", "n", "accuracy", "precision", "recall", "f1")
    rows = []
    for split in bundle.all_splits():
        preds = model.predict_texts(list(split.texts)).tolist()
        overall = binary_scores(split.labels, preds)
        rows.append(
            (
                split.name,
                "all",
                overall.n,
                overall.accuracy,
                overall.precision,
                overall.recall,
                overall.f1,
            )
        )
        for slice_row in slice_scores(list(split.texts), list(split.labels), preds):
            if slice_row.name == "all" or slice_row.scores is None:
                continue
            s = slice_row.scores
            rows.append(
                (
                    split.name,
                    slice_row.name,
                    slice_row.n,
                    s.accuracy,
                    s.precision,
                    s.recall,
                    s.f1,
                )
            )
    print(markdown_table(headers, rows))
    print()
    print(
        "This is not the 2023 SVM. If `not_tag` / `explicit` dominate the "
        "cue weights and the no_explicit_cue slice drops, the official "
        "test number is partly hashtag leakage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
