#!/usr/bin/env python3
"""Train a cue-only logistic model on the real CSVs (NumPy, no sklearn)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset_io import load_all_splits
from examples.lib.lexical_features import FEATURE_NAMES, cue_presence_table, featurize_corpus
from examples.lib.logistic import LogisticBinary, binary_metrics


def _fmt_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} "
        f"| {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
    )


def render(
    cue_table: dict[str, dict[str, float]],
    train_m: dict[str, float],
    test_m: dict[str, float],
    sub_m: dict[str, float],
    weights: np.ndarray,
) -> str:
    ranked = sorted(
        ((name, float(weights[FEATURE_NAMES.index(name)])) for name in FEATURE_NAMES if name != "bias"),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    lines = [
        "# Lexical baseline",
        "",
        "L2 logistic regression on hand-built cues from `examples/lib/lexical_features.py`.",
        "Trained on train, evaluated on test and subtest. No embeddings.",
        "",
        "## Metrics",
        "",
        "| Split | accuracy | precision | recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        _fmt_metrics("train", train_m),
        _fmt_metrics("test", test_m),
        _fmt_metrics("subtest", sub_m),
        "",
        "## Cue rates on train (binary features)",
        "",
        "| feature | all | sarcastic | non-sarcastic |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in sorted(cue_table.items(), key=lambda kv: kv[1]["sarcastic"], reverse=True):
        lines.append(
            f"| `{name}` | {row['all']:.3f} | {row['sarcastic']:.3f} "
            f"| {row['non_sarcastic']:.3f} |"
        )
    lines += [
        "",
        "## Largest |weight| after standardization",
        "",
        "| feature | weight |",
        "| --- | ---: |",
    ]
    for name, weight in ranked[:16]:
        lines.append(f"| `{name}` | {weight:+.3f} |")
    lines += [
        "",
        "Positive weight pushes the sarcastic class. `#not` / sarcasm hashtags",
        "should dominate; emoji faces are weaker because they also appear on",
        "sincere tweets.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    splits = load_all_splits(args.root)
    X_train = featurize_corpus(splits["train"].texts)
    y_train = np.asarray(splits["train"].labels, dtype=np.int32)
    X_test = featurize_corpus(splits["test"].texts)
    y_test = np.asarray(splits["test"].labels, dtype=np.int32)
    X_sub = featurize_corpus(splits["subtest"].texts)
    y_sub = np.asarray(splits["subtest"].labels, dtype=np.int32)

    model = LogisticBinary(epochs=args.epochs, seed=args.seed).fit(X_train, y_train)
    train_m = binary_metrics(y_train, model.predict(X_train))
    test_m = binary_metrics(y_test, model.predict(X_test))
    sub_m = binary_metrics(y_sub, model.predict(X_sub))
    cues = cue_presence_table(splits["train"].texts, splits["train"].labels)
    report = render(cues, train_m, test_m, sub_m, model.weights_)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
