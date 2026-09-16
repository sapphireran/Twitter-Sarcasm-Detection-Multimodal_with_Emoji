"""Binary classification metrics with no numpy / sklearn dependency."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def _as_lists(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[List[int], List[int]]:
    gold = list(y_true)
    pred = list(y_pred)
    if len(gold) != len(pred):
        raise ValueError(f"length mismatch: {len(gold)} gold vs {len(pred)} pred")
    return gold, pred


def confusion(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` for the positive class ``1``."""
    gold, pred = _as_lists(y_true, y_pred)
    tp = fp = tn = fn = 0
    for t, p in zip(gold, pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    gold, pred = _as_lists(y_true, y_pred)
    if not gold:
        return 0.0
    return sum(int(t == p) for t, p in zip(gold, pred)) / len(gold)


def precision(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    tp, fp, _tn, _fn = confusion(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom else 0.0


def recall(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    tp, _fp, _tn, fn = confusion(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom else 0.0


def f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return 2 * p * r / denom if denom else 0.0


def majority_baseline(y_true: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    ones = sum(y_true)
    zeros = len(y_true) - ones
    return max(ones, zeros) / len(y_true)
