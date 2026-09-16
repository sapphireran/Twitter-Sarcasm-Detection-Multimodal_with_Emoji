"""Tiny logistic regression over lexical sarcasm cues.

This is not a replacement for the Bi-LSTM + attention models. It exists so the
examples can train *something* on the checked-in CSVs without GloVe, emoji2vec,
or TensorFlow. The learned weights are also a useful sanity check: on this
dataset, ``#not`` and contrast cues should dominate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.lib.tweet_features import FEATURE_NAMES, extract_features


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def featurize(texts: list[str]) -> np.ndarray:
    return np.asarray([extract_features(text).as_vector() for text in texts], dtype=np.float64)


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)

    def decision(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision(features))

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(int)

    def top_weights(self, k: int = 8) -> list[tuple[str, float]]:
        order = np.argsort(np.abs(self.weights))[::-1]
        return [(self.feature_names[i], float(self.weights[i])) for i in order[:k]]


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.25,
    epochs: int = 250,
    l2: float = 1e-3,
    seed: int = 0,
) -> LogisticModel:
    """Batch gradient descent for binary logistic regression."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=features.shape[1])
    bias = 0.0
    y = labels.astype(np.float64)
    n = float(len(y))
    for _ in range(epochs):
        probs = sigmoid(features @ weights + bias)
        error = probs - y
        weights -= learning_rate * ((features.T @ error) / n + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return LogisticModel(weights=weights, bias=float(bias))


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n: int
    positives: int

    def as_percent_row(self) -> dict[str, str]:
        return {
            "n": str(self.n),
            "positives": str(self.positives),
            "accuracy": f"{100 * self.accuracy:.2f}",
            "precision": f"{100 * self.precision:.2f}",
            "recall": f"{100 * self.recall:.2f}",
            "f1": f"{100 * self.f1:.2f}",
        }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(
        accuracy=float(np.mean(y_true == y_pred)),
        precision=precision,
        recall=recall,
        f1=f1,
        n=int(len(y_true)),
        positives=int(np.sum(y_true == 1)),
    )
