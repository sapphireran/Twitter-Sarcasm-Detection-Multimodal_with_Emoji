"""Cue-based baselines that run without scikit-learn or TensorFlow.

``LexiconBaseline`` is the honest floor for this dataset: predict
sarcastic if any self-annotation hashtag from ``CUE_HASHTAGS`` is present.

``CueLogistic`` is a from-scratch logistic regressor on the cue vector.
It is a teaching model, not a replacement for the 2023 Random Forest or
BiLSTM. Training uses batch gradient descent on a standardized feature
matrix so the walkthrough stays deterministic and dependency-free.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .cues import extract_cue_features, extract_matrix, feature_names
from .tokenize import tokenize_tweet

DEFAULT_SEED = 2023


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _standardize(matrix: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    if not matrix:
        return [], [], []
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    means = [0.0] * n_cols
    for row in matrix:
        for idx, value in enumerate(row):
            means[idx] += value
    means = [value / n_rows for value in means]
    variances = [0.0] * n_cols
    for row in matrix:
        for idx, value in enumerate(row):
            variances[idx] += (value - means[idx]) ** 2
    stds = [math.sqrt(value / n_rows) if value > 0 else 1.0 for value in variances]
    scaled = [
        [(value - means[idx]) / stds[idx] for idx, value in enumerate(row)]
        for row in matrix
    ]
    return scaled, means, stds


def _apply_standardize(
    matrix: list[list[float]], means: Sequence[float], stds: Sequence[float]
) -> list[list[float]]:
    return [
        [(value - means[idx]) / stds[idx] for idx, value in enumerate(row)]
        for row in matrix
    ]


class LexiconBaseline:
    """Predict sarcastic iff a known cue hashtag is present."""

    name = "lexicon_hashtag"

    def predict_one(self, text: str) -> int:
        tokens = tokenize_tweet(text)
        from .cues import CUE_HASHTAGS
        from .tokenize import hashtags

        return int(bool(set(hashtags(tokens)) & CUE_HASHTAGS))

    def predict(self, texts: Sequence[str]) -> list[int]:
        return [self.predict_one(text) for text in texts]


@dataclass
class CueLogistic:
    """Binary logistic regression on :func:`extract_cue_features`."""

    learning_rate: float = 0.25
    epochs: int = 40
    l2: float = 0.01
    seed: int = DEFAULT_SEED
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    feature_names_: tuple[str, ...] = field(default_factory=feature_names)

    name = "cue_logistic"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "CueLogistic":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        rng = random.Random(self.seed)
        order = list(range(len(texts)))
        rng.shuffle(order)
        raw = extract_matrix(texts)
        scaled, self.means, self.stds = _standardize([raw[i] for i in order])
        y = [labels[i] for i in order]
        n_features = len(scaled[0]) if scaled else 0
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(scaled) or 1
        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for row, label in zip(scaled, y):
                pred = _sigmoid(self._logit_from_scaled(row))
                error = pred - label
                for idx, value in enumerate(row):
                    grad_w[idx] += error * value
                grad_b += error
            for idx in range(n_features):
                grad_w[idx] = grad_w[idx] / n + self.l2 * self.weights[idx]
                self.weights[idx] -= self.learning_rate * grad_w[idx]
            self.bias -= self.learning_rate * (grad_b / n)
        return self

    def _logit_from_scaled(self, row: Sequence[float]) -> float:
        return self.bias + sum(w * x for w, x in zip(self.weights, row))

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        if not self.weights:
            raise RuntimeError("CueLogistic.fit must be called first")
        raw = extract_matrix(texts)
        scaled = _apply_standardize(raw, self.means, self.stds)
        return [_sigmoid(self._logit_from_scaled(row)) for row in scaled]

    def predict(self, texts: Sequence[str], threshold: float = 0.5) -> list[int]:
        return [int(prob >= threshold) for prob in self.predict_proba(texts)]

    def top_weights(self, k: int = 8) -> list[tuple[str, float]]:
        pairs = list(zip(self.feature_names_, self.weights))
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        return pairs[:k]


def describe_prediction(text: str, model: CueLogistic | LexiconBaseline) -> dict:
    """Bundle tokens, cues, and the model decision for walkthroughs."""
    from .cues import cue_hashtags_in, extract_cue_features

    tokens = tokenize_tweet(text)
    features = extract_cue_features(text, tokens=tokens)
    if isinstance(model, CueLogistic):
        proba = model.predict_proba([text])[0]
        label = int(proba >= 0.5)
    else:
        proba = float(model.predict_one(text))
        label = int(proba)
    return {
        "text": text,
        "tokens": tokens,
        "cue_hashtags": cue_hashtags_in(text),
        "features": dict(zip(feature_names(), features.as_list())),
        "probability": proba,
        "prediction": label,
    }
