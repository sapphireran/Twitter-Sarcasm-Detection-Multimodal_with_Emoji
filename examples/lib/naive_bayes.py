"""Bernoulli Naive Bayes over 0/1 feature dicts.

This is the classifier used by ``examples/heuristic_baseline.py``. It is small
enough to read in one sitting and has no sklearn dependency.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .features import FEATURE_NAMES


class BernoulliNB:
    """Binary-feature Naive Bayes with Laplace smoothing."""

    def __init__(self, feature_names: Sequence[str] = FEATURE_NAMES, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.feature_names = tuple(feature_names)
        self.alpha = float(alpha)
        self.class_log_prior: Dict[int, float] = {}
        self.feature_log_prob: Dict[int, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, rows: Iterable[Dict[str, int]], labels: Iterable[int]) -> "BernoulliNB":
        X = list(rows)
        y = list(labels)
        if len(X) != len(y):
            raise ValueError("rows and labels have different lengths")
        if not X:
            raise ValueError("empty training set")

        counts = {0: 0, 1: 0}
        feat_ones = {
            0: {name: 0 for name in self.feature_names},
            1: {name: 0 for name in self.feature_names},
        }
        for row, label in zip(X, y):
            if label not in (0, 1):
                raise ValueError(f"labels must be 0 or 1, got {label!r}")
            counts[label] += 1
            for name in self.feature_names:
                if row.get(name, 0):
                    feat_ones[label][name] += 1

        n = counts[0] + counts[1]
        self.class_log_prior = {
            0: math.log((counts[0] + self.alpha) / (n + 2 * self.alpha)),
            1: math.log((counts[1] + self.alpha) / (n + 2 * self.alpha)),
        }
        self.feature_log_prob = {0: {}, 1: {}}
        for label in (0, 1):
            # Laplace: (ones + alpha) / (class_count + 2 alpha) because Bernoulli.
            denom = counts[label] + 2 * self.alpha
            for name in self.feature_names:
                ones = feat_ones[label][name] + self.alpha
                self.feature_log_prob[label][name] = math.log(ones / denom)
        self._fitted = True
        return self

    def _log_prob(self, row: Dict[str, int], label: int) -> float:
        score = self.class_log_prior[label]
        for name in self.feature_names:
            log_p = self.feature_log_prob[label][name]
            # P(x=1) stored; P(x=0) = 1 - P(x=1).
            p = math.exp(log_p)
            p = min(max(p, 1e-12), 1.0 - 1e-12)
            if row.get(name, 0):
                score += math.log(p)
            else:
                score += math.log(1.0 - p)
        return score

    def predict_one(self, row: Dict[str, int]) -> int:
        if not self._fitted:
            raise RuntimeError("BernoulliNB.fit() must be called first")
        return 1 if self._log_prob(row, 1) >= self._log_prob(row, 0) else 0

    def predict(self, rows: Iterable[Dict[str, int]]) -> List[int]:
        return [self.predict_one(row) for row in rows]

    def predict_log_odds(self, row: Dict[str, int]) -> float:
        """log P(y=1|x) − log P(y=0|x) up to the shared evidence term."""
        if not self._fitted:
            raise RuntimeError("BernoulliNB.fit() must be called first")
        return self._log_prob(row, 1) - self._log_prob(row, 0)

    def debug_top_features(self, k: int = 5) -> List[Tuple[str, float]]:
        """Features with the largest log-odds for the sarcastic class when on."""
        if not self._fitted:
            raise RuntimeError("BernoulliNB.fit() must be called first")
        scored = []
        for name in self.feature_names:
            p1 = math.exp(self.feature_log_prob[1][name])
            p0 = math.exp(self.feature_log_prob[0][name])
            scored.append((name, math.log(p1) - math.log(p0)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
