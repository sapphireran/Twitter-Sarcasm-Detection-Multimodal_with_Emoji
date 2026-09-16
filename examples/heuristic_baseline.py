#!/usr/bin/env python3
"""Hashtag-only sarcasm rule, scored on the checked-in splits.

This is a pedagogical ceiling, not a 2023 notebook result. It answers:
how far do you get if you only trust #not / #sarcasm / #sarcastic(tweet)?

    python examples/heuristic_baseline.py
    python examples/heuristic_baseline.py --show-errors --split test --n-errors 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset import load_all  # noqa: E402
from examples.lib.tokenize import tokenize_tweet  # noqa: E402

SARC_TAGS = {"#not", "#sarcasm", "#sarcastic", "#sarcastictweet"}


def predict(sentence: str) -> int:
    tags = {tok for tok in tokenize_tweet(sentence) if tok.startswith("#")}
    return int(bool(tags & SARC_TAGS))


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for gold, pred in zip(y_true, y_pred):
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    c = confusion(y_true, y_pred)
    acc = safe_div(c["tp"] + c["tn"], len(y_true))
    prec = safe_div(c["tp"], c["tp"] + c["fp"])
    rec = safe_div(c["tp"], c["tp"] + c["fn"])
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {
        **{k: float(v) for k, v in c.items()},
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def collect_errors(
    sentences: List[str], labels: List[int], kind: str, limit: int
) -> List[Tuple[str, int, int]]:
    out = []
    for sentence, gold in zip(sentences, labels):
        pred = predict(sentence)
        if kind == "fp" and pred == 1 and gold == 0:
            out.append((sentence, gold, pred))
        elif kind == "fn" and pred == 0 and gold == 1:
            out.append((sentence, gold, pred))
        if len(out) >= limit:
            break
    return out


def _print_metrics(name: str, block: Dict[str, float]) -> None:
    print(f"=== {name} ===")
    print(
        f"accuracy {block['accuracy']:.4f}   "
        f"precision {block['precision']:.4f}   "
        f"recall {block['recall']:.4f}   "
        f"f1 {block['f1']:.4f}"
    )
    print(
        f"tp {int(block['tp'])}  fp {int(block['fp'])}  "
        f"tn {int(block['tn'])}  fn {int(block['fn'])}"
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", choices=("train", "test", "subtest"))
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--n-errors", type=int, default=5)
    args = parser.parse_args()

    print(
        "Rule: predict sarcastic iff the walkthrough tokenizer emits "
        f"one of {sorted(SARC_TAGS)}.\n"
        "Compare these F1 numbers to Bi-LSTM+ATT test WE F1 = 0.8686 "
        "in docs/results.md.\n"
    )
    splits = load_all()
    names = args.split or list(splits)
    for name in names:
        split = splits[name]
        preds = [predict(s) for s in split.sentences]
        block = metrics(split.labels, preds)
        _print_metrics(name, block)
        if args.show_errors:
            print(f"false positives on {name}:")
            fps = collect_errors(split.sentences, split.labels, "fp", args.n_errors)
            if not fps:
                print("  (none in this window)")
            for sentence, gold, pred in fps:
                print(f"  gold={gold} pred={pred}  {sentence[:160]}")
            print(f"false negatives on {name}:")
            fns = collect_errors(split.sentences, split.labels, "fn", args.n_errors)
            for sentence, gold, pred in fns:
                print(f"  gold={gold} pred={pred}  {sentence[:160]}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
