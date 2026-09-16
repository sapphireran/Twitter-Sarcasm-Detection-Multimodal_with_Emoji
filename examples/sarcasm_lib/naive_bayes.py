"""Multinomial Naive Bayes with Laplace smoothing, from scratch."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .features import lexical_tokens


@dataclass
class MultinomialNB:
    alpha: float = 1.0
    strip_supervision_tags: bool = False
    include_cues: bool = True
    min_count: int = 2
    class_log_prior_: np.ndarray = field(init=False, repr=False)
    feature_log_prob_: np.ndarray = field(init=False, repr=False)
    vocab_: dict[str, int] = field(init=False, repr=False)
    token_index_: list[str] = field(init=False, repr=False)

    def _extract(self, text: str) -> list[str]:
        return lexical_tokens(
            text,
            strip_tags=self.strip_supervision_tags,
            include_cues=self.include_cues,
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "MultinomialNB":
        if len(texts) != len(labels):
            raise ValueError("texts and labels length mismatch")
        df: Counter[str] = Counter()
        per_class: list[Counter[str]] = [Counter(), Counter()]
        n_class = [0, 0]
        for text, label in zip(texts, labels):
            if label not in (0, 1):
                raise ValueError(f"expected binary label, got {label}")
            tokens = self._extract(text)
            n_class[label] += 1
            per_class[label].update(tokens)
            df.update(set(tokens))
        vocab_tokens = [tok for tok, count in df.items() if count >= self.min_count]
        vocab_tokens.sort()
        self.token_index_ = vocab_tokens
        self.vocab_ = {tok: i for i, tok in enumerate(vocab_tokens)}
        n_features = len(vocab_tokens)
        counts = np.zeros((2, n_features), dtype=np.float64)
        for label in (0, 1):
            for token, freq in per_class[label].items():
                index = self.vocab_.get(token)
                if index is not None:
                    counts[label, index] = freq
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        total = sum(n_class)
        self.class_log_prior_ = np.log(np.array(n_class, dtype=np.float64) / total)
        return self

    def _vectorize(self, texts: Iterable[str]) -> np.ndarray:
        matrix = np.zeros((0, len(self.vocab_)), dtype=np.float64)
        rows = []
        for text in texts:
            row = np.zeros(len(self.vocab_), dtype=np.float64)
            for token in self._extract(text):
                index = self.vocab_.get(token)
                if index is not None:
                    row[index] += 1.0
            rows.append(row)
        if not rows:
            return matrix
        return np.vstack(rows)

    def predict_log_proba(self, texts: Sequence[str]) -> np.ndarray:
        counts = self._vectorize(texts)
        return counts @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        log_prob = self.predict_log_proba(texts)
        log_prob = log_prob - log_prob.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        return prob / prob.sum(axis=1, keepdims=True)

    def predict(self, texts: Sequence[str]) -> list[int]:
        log_prob = self.predict_log_proba(texts)
        return np.argmax(log_prob, axis=1).astype(int).tolist()

    def top_tokens(self, label: int, k: int = 15) -> list[tuple[str, float]]:
        """Tokens with the largest log P(token | label) − log P(token | other)."""
        other = 1 - label
        delta = self.feature_log_prob_[label] - self.feature_log_prob_[other]
        order = np.argsort(delta)[::-1][:k]
        return [(self.token_index_[i], float(delta[i])) for i in order]
