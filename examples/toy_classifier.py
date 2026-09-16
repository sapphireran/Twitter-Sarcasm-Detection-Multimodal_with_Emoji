"""A numpy logistic regressor so the toy corpus can be classified without sklearn.

This is not one of the 2023 baselines. It exists so ``run_toy_classifier.py``
can show that concatenating the emoji channel changes the decision boundary
on a 16-tweet set you can read in one screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogReg:
    weights: np.ndarray  # (features,)
    bias: float
    losses: list[float]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.05,
    lr: float = 0.35,
    steps: int = 400,
    seed: int = 0,
) -> LogReg:
    if X.ndim != 2:
        raise ValueError("X must be 2-d")
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=(X.shape[1],))
    bias = 0.0
    losses: list[float] = []
    n = float(len(y))
    for _ in range(steps):
        logits = X @ weights + bias
        probs = _sigmoid(logits)
        # Mean binary cross-entropy + L2 on weights (not bias).
        loss = float(
            -np.mean(y * np.log(probs + 1e-9) + (1.0 - y) * np.log(1.0 - probs + 1e-9))
            + 0.5 * l2 * float(np.dot(weights, weights))
        )
        losses.append(loss)
        err = (probs - y) / n
        weights = weights - lr * (X.T @ err + l2 * weights)
        bias = bias - lr * float(np.sum(err))
    return LogReg(weights=weights, bias=bias, losses=losses)


def predict_proba(model: LogReg, X: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ model.weights + model.bias)


def predict(model: LogReg, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(model, X) >= threshold).astype(int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
