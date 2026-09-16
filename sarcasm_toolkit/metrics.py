"""Binary classification metrics used in the 2023 course report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    support_positive: int
    support_negative: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
            "support_positive": self.support_positive,
            "support_negative": self.support_negative,
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
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
    return tp, fp, tn, fn


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> BinaryMetrics:
    tp, fp, tn, fn = confusion(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    total = tp + fp + tn + fn
    return BinaryMetrics(
        accuracy=_safe_div(tp + tn, total),
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        support_positive=tp + fn,
        support_negative=tn + fp,
    )


def format_metrics(name: str, metrics: BinaryMetrics) -> str:
    return (
        f"{name:24s}  acc={metrics.accuracy:.4f}  "
        f"p={metrics.precision:.4f}  r={metrics.recall:.4f}  "
        f"f1={metrics.f1:.4f}  "
        f"tp={metrics.true_positive} fp={metrics.false_positive} "
        f"tn={metrics.true_negative} fn={metrics.false_negative}"
    )


def majority_baseline(y_true: Iterable[int]) -> list[int]:
    labels = list(y_true)
    ones = sum(labels)
    majority = 1 if ones >= (len(labels) - ones) else 0
    return [majority] * len(labels)
