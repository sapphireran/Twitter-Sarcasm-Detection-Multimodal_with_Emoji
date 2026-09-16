"""Binary classification scores and small-sample intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BinaryScores:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        den = self.tp + self.fp
        return self.tp / den if den else 0.0

    @property
    def recall(self) -> float:
        den = self.tp + self.fn
        return self.tp / den if den else 0.0

    @property
    def f1(self) -> float:
        den = self.precision + self.recall
        return 2 * self.precision * self.recall / den if den else 0.0

    @property
    def specificity(self) -> float:
        den = self.tn + self.fp
        return self.tn / den if den else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
        }


def binary_scores(y_true: Sequence[int], y_pred: Sequence[int]) -> BinaryScores:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if pred not in (0, 1) or truth not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 0:
            tn += 1
        else:
            fn += 1
    return BinaryScores(n=len(y_true), tp=tp, fp=fp, tn=tn, fn=fn)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return ``(p, lo, hi)`` for ``k`` successes out of ``n`` trials."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def odds_ratio(hit_pos: int, hit_neg: int, miss_pos: int, miss_neg: int, *, smooth: float = 0.5) -> float:
    """Haldane-Anscombe corrected odds ratio."""
    a = hit_pos + smooth
    b = hit_neg + smooth
    c = miss_pos + smooth
    d = miss_neg + smooth
    return (a / b) / (c / d)


def mean(values: Iterable[float]) -> float:
    seq = list(values)
    if not seq:
        return 0.0
    return sum(seq) / len(seq)
