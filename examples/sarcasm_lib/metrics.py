"""Binary classification metrics without scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Confusion:
    true_neg: int
    false_pos: int
    false_neg: int
    true_pos: int

    @property
    def n(self) -> int:
        return self.true_neg + self.false_pos + self.false_neg + self.true_pos


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Confusion:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    tn = fp = fn = tp = 0
    for truth, pred in zip(y_true, y_pred):
        if truth not in (0, 1) or pred not in (0, 1):
            raise ValueError(f"expected binary labels, got {truth}, {pred}")
        if truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    return Confusion(tn, fp, fn, tp)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    table = confusion(y_true, y_pred)
    return (table.true_pos + table.true_neg) / table.n if table.n else 0.0


def precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    table = confusion(y_true, y_pred)
    denom = table.true_pos + table.false_pos
    return table.true_pos / denom if denom else 0.0


def recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    table = confusion(y_true, y_pred)
    denom = table.true_pos + table.false_neg
    return table.true_pos / denom if denom else 0.0


def binary_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def format_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    table = confusion(y_true, y_pred)
    return (
        f"n={table.n}  acc={accuracy(y_true, y_pred):.4f}  "
        f"prec={precision(y_true, y_pred):.4f}  "
        f"rec={recall(y_true, y_pred):.4f}  "
        f"f1={binary_f1(y_true, y_pred):.4f}  "
        f"tp={table.true_pos} fp={table.false_pos} "
        f"tn={table.true_neg} fn={table.false_neg}"
    )
