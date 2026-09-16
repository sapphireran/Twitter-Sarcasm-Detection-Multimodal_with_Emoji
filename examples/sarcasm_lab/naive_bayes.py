"""Multinomial Naive Bayes over sparse count dicts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MultinomialNB:
    alpha: float = 1.0
    class_log_prior_: list[float] = field(default_factory=list)
    feature_log_prob_: list[list[float]] = field(default_factory=list)
    classes_: list[int] = field(default_factory=list)

    def fit(self, X: list[dict[int, int]], y: list[int], n_features: int) -> MultinomialNB:
        if len(X) != len(y):
            raise ValueError("X/y length mismatch")
        classes = sorted(set(y))
        if classes != [0, 1] and classes != [0] and classes != [1]:
            # Still allow other integer labels, but keep them ordered.
            pass
        n_classes = len(classes)
        class_index = {label: i for i, label in enumerate(classes)}
        class_count = [0] * n_classes
        feature_count = [[0.0] * n_features for _ in range(n_classes)]
        for row, label in zip(X, y):
            ci = class_index[label]
            class_count[ci] += 1
            for idx, value in row.items():
                feature_count[ci][idx] += value

        total = float(sum(class_count))
        self.classes_ = classes
        self.class_log_prior_ = [math.log(c / total) if c else -math.inf for c in class_count]
        self.feature_log_prob_ = []
        for ci in range(n_classes):
            summed = sum(feature_count[ci]) + self.alpha * n_features
            log_probs = [
                math.log((feature_count[ci][j] + self.alpha) / summed) for j in range(n_features)
            ]
            self.feature_log_prob_.append(log_probs)
        return self

    def _log_proba_row(self, row: dict[int, int]) -> list[float]:
        scores = list(self.class_log_prior_)
        for ci, log_probs in enumerate(self.feature_log_prob_):
            acc = scores[ci]
            for idx, value in row.items():
                acc += value * log_probs[idx]
            scores[ci] = acc
        return scores

    def predict(self, X: list[dict[int, int]]) -> list[int]:
        preds: list[int] = []
        for row in X:
            scores = self._log_proba_row(row)
            best = max(range(len(scores)), key=lambda i: scores[i])
            preds.append(self.classes_[best])
        return preds
