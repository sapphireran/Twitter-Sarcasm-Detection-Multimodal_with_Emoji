"""From-scratch binary logistic regression (NumPy only)."""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionGD:
    def __init__(
        self,
        lr: float = 0.15,
        epochs: int = 250,
        l2: float = 0.002,
        seed: int = 0,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"bad shapes x={x.shape} y={y.shape}")
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.05, size=(x.shape[1],))
        # Keep the bias (col 0) close to the class prior.
        prior = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        w[0] = np.log(prior / (1.0 - prior))
        n = float(x.shape[0])
        self.loss_history = []
        for _ in range(self.epochs):
            pred = _sigmoid(x @ w)
            error = pred - y
            grad = (x.T @ error) / n
            # Do not L2-penalize the bias term.
            penalty = self.l2 * np.concatenate([[0.0], w[1:]])
            w = w - self.lr * (grad + penalty)
            loss = float(
                -np.mean(y * np.log(pred + 1e-9) + (1 - y) * np.log(1 - pred + 1e-9))
            )
            self.loss_history.append(loss)
        self.weights = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("call fit() first")
        return _sigmoid(np.asarray(x, dtype=np.float64) @ self.weights)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(np.int64)
