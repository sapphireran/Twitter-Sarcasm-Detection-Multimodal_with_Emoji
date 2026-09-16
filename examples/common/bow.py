"""Multinomial Naive Bayes over bag-of-tokens.

Used by ``06_bow_baseline.py`` as a no-TensorFlow, no-sklearn reference
classifier. Laplace smoothing matches the usual text-classification default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class CountVectorizer:
    vocab: Dict[str, int]
    idf: np.ndarray | None = None

    @classmethod
    def fit(cls, docs: Sequence[Sequence[str]], min_count: int = 2, max_features: int = 4000) -> "CountVectorizer":
        counts: Dict[str, int] = {}
        for doc in docs:
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
        items = [(tok, n) for tok, n in counts.items() if n >= min_count]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        items = items[:max_features]
        vocab = {tok: i for i, (tok, _) in enumerate(items)}
        df = np.zeros(len(vocab), dtype=np.int32)
        for doc in docs:
            seen = {tok for tok in doc if tok in vocab}
            for tok in seen:
                df[vocab[tok]] += 1
        n_docs = max(len(docs), 1)
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return cls(vocab=vocab, idf=idf)

    def transform(self, docs: Sequence[Sequence[str]], tfidf: bool = False) -> np.ndarray:
        x = np.zeros((len(docs), len(self.vocab)), dtype=np.float64)
        for i, doc in enumerate(docs):
            for tok in doc:
                j = self.vocab.get(tok)
                if j is not None:
                    x[i, j] += 1.0
        if tfidf:
            if self.idf is None:
                raise ValueError("vectorizer has no idf")
            x = x * self.idf
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            x = x / norms
        return x


@dataclass
class MultinomialNB:
    log_prior: np.ndarray
    log_lik: np.ndarray  # (n_classes, n_features)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> "MultinomialNB":
        classes = np.array([0, 1])
        n_classes = 2
        n_features = x.shape[1]
        class_count = np.array([(y == c).sum() for c in classes], dtype=np.float64)
        log_prior = np.log(class_count + alpha) - np.log(class_count.sum() + n_classes * alpha)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        for c in classes:
            feature_count[c] = x[y == c].sum(axis=0)
        smoothed = feature_count + alpha
        log_lik = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        return cls(log_prior=log_prior, log_lik=log_lik)

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        return self.log_prior + x @ self.log_lik.T

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_log_proba(x), axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)
