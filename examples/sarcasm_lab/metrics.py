"""Binary classification metrics without scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def as_percent_row(self) -> dict[str, str]:
        return {
            "n": str(self.n),
            "acc": f"{100 * self.accuracy:.2f}",
            "p": f"{100 * self.precision:.2f}",
            "r": f"{100 * self.recall:.2f}",
            "f1": f"{100 * self.f1:.2f}",
        }


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def binary_metrics(y_true: list[int], y_pred: list[int]) -> BinaryMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    tp = fp = tn = fn = 0
    for gold, pred in zip(y_true, y_pred):
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, len(y_true))
    return BinaryMetrics(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        n=len(y_true),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def format_metrics(name: str, metrics: BinaryMetrics) -> str:
    row = metrics.as_percent_row()
    return (
        f"{name:28s}  n={row['n']:>5}  "
        f"acc={row['acc']}%  p={row['p']}%  r={row['r']}%  f1={row['f1']}%"
    )
