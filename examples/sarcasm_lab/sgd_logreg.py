"""L2-regularized logistic regression trained with SGD on sparse rows."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class SGDLogisticRegression:
    n_features: int = 0
    lr: float = 0.05
    l2: float = 1e-4
    epochs: int = 4
    seed: int = 0
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0

    def fit(self, X: list[dict[int, int]], y: list[int], n_features: int) -> SGDLogisticRegression:
        if len(X) != len(y):
            raise ValueError("X/y length mismatch")
        self.n_features = n_features
        rng = random.Random(self.seed)
        self.weights = [0.0] * n_features
        self.bias = 0.0
        indices = list(range(len(X)))
        for _ in range(self.epochs):
            rng.shuffle(indices)
            for i in indices:
                row = X[i]
                gold = float(y[i])
                z = self.bias
                for idx, value in row.items():
                    z += self.weights[idx] * value
                pred = _sigmoid(z)
                err = pred - gold
                # Decay then add the sparse gradient. Dense L2 on untouched
                # weights is skipped for speed; only seen features are decayed.
                for idx, value in row.items():
                    self.weights[idx] -= self.lr * (err * value + self.l2 * self.weights[idx])
                self.bias -= self.lr * err
        return self

    def predict_proba(self, X: list[dict[int, int]]) -> list[float]:
        probs: list[float] = []
        for row in X:
            z = self.bias
            for idx, value in row.items():
                z += self.weights[idx] * value
            probs.append(_sigmoid(z))
        return probs

    def predict(self, X: list[dict[int, int]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]
