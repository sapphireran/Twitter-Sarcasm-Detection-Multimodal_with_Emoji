"""Tiny classifiers for the toy multi-modal pipeline.

The 2023 notebooks train sklearn SVM / trees and a Keras BiLSTM. The
examples only need something that can show *why* concatenating emoji
means changes a decision. Logistic regression and a nearest-centroid
rule are enough, and they stay in NumPy.
"""

from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logreg(
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float = 0.4,
    steps: int = 600,
    l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Fit a binary logistic model with batch gradient descent.

    Returns ``(weight, bias)``. Labels must be 0/1.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-d, got {x.shape}")
    if y.shape != (x.shape[0],):
        raise ValueError("labels must be a vector aligned with features")
    weight = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    n = float(x.shape[0])
    for _ in range(steps):
        pred = sigmoid(x @ weight + bias)
        err = pred - y
        weight -= learning_rate * ((x.T @ err) / n + l2 * weight)
        bias -= learning_rate * (err.mean())
    return weight, bias


def predict_proba(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(np.asarray(features, dtype=np.float64) @ weight + bias)


def predict_label(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    return (predict_proba(features, weight, bias) >= 0.5).astype(int)


def nearest_centroid(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean vector of class 0 and class 1."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("nearest_centroid needs both classes present")
    return x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)


def predict_nearest_centroid(
    features: np.ndarray,
    centroid0: np.ndarray,
    centroid1: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    d0 = np.linalg.norm(x - centroid0, axis=1)
    d1 = np.linalg.norm(x - centroid1, axis=1)
    return (d1 < d0).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot score an empty prediction")
    return float(np.mean(y_true == y_pred))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fp = np.sum((y_true != positive) & (y_pred == positive))
    fn = np.sum((y_true == positive) & (y_pred != positive))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))
