"""Minimal binary logistic regression (L2, mini-batch SGD).

Used by examples/lexical_baseline.py so the floor model has no sklearn
dependency. This is not a general-purpose solver.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticBinary:
    def __init__(
        self,
        l2: float = 1e-3,
        lr: float = 0.05,
        epochs: int = 12,
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticBinary:
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        # Standardize non-bias columns so SGD step size is sane.
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._mean[0] = 0.0
        self._std[0] = 1.0
        self._std[self._std < 1e-8] = 1.0
        Xs = (X - self._mean) / self._std

        w = rng.normal(scale=0.01, size=(d,))
        y = y.astype(np.float64)
        order = np.arange(n)
        for _ in range(self.epochs):
            rng.shuffle(order)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = Xs[idx]
                pred = _sigmoid(xb @ w)
                err = pred - y[idx]
                grad = (xb.T @ err) / len(idx) + self.l2 * w
                grad[0] -= self.l2 * w[0]  # do not shrink the bias
                w -= self.lr * grad
        self.weights_ = w
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("call fit() first")
        return _sigmoid(self._transform(X) @ self.weights_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
