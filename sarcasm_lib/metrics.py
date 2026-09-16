"""Binary classification metrics without scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def n(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    counts: ConfusionCounts


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionCounts:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred, strict=True):
        if truth not in (0, 1) or pred not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return ConfusionCounts(tp, fp, tn, fn)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> BinaryMetrics:
    counts = confusion_counts(y_true, y_pred)
    accuracy = _safe_div(counts.true_positive + counts.true_negative, counts.n)
    precision = _safe_div(counts.true_positive, counts.true_positive + counts.false_positive)
    recall = _safe_div(counts.true_positive, counts.true_positive + counts.false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return BinaryMetrics(accuracy, precision, recall, f1, counts)
