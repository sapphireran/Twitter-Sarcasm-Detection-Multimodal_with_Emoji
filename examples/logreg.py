"""A small L2 logistic regressor so the lexical example needs no sklearn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticReport:
    weights: np.ndarray
    bias: float
    train_loss: float
    n_epochs: int


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logreg(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-2,
    lr: float = 0.15,
    epochs: int = 250,
    seed: int = 7,
) -> LogisticReport:
    if x.ndim != 2:
        raise ValueError("x must be a 2-d feature matrix")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y length must match x rows")
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.01, size=x.shape[1])
    bias = 0.0
    y = y.astype(np.float64)
    n = max(x.shape[0], 1)
    last_loss = 0.0
    for _ in range(epochs):
        logits = x @ weights + bias
        probs = _sigmoid(logits)
        error = probs - y
        grad_w = (x.T @ error) / n + l2 * weights
        grad_b = float(error.mean())
        weights = weights - lr * grad_w
        bias = bias - lr * grad_b
        last_loss = float(
            -np.mean(y * np.log(probs + 1e-9) + (1.0 - y) * np.log(1.0 - probs + 1e-9))
            + 0.5 * l2 * float(np.dot(weights, weights))
        )
    return LogisticReport(
        weights=weights, bias=float(bias), train_loss=last_loss, n_epochs=epochs
    )


def predict_proba(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return _sigmoid(x @ weights + bias)


def predict_label(
    x: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5
) -> np.ndarray:
    return (predict_proba(x, weights, bias) >= threshold).astype(int)


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def format_scores(title: str, scores: dict) -> str:
    return (
        f"{title}: acc {scores['accuracy']:.4f}  "
        f"f1 {scores['f1']:.4f}  "
        f"prec {scores['precision']:.4f}  "
        f"rec {scores['recall']:.4f}  "
        f"(tp {scores['tp']} fp {scores['fp']} fn {scores['fn']} tn {scores['tn']})"
    )
