"""Hashed unigram TF-IDF and sparse logistic regression.

The example baseline avoids scikit-learn and a dense 40k-by-vocab matrix.
Tokens are hashed into a fixed-width bag (the hashing trick), scaled by a
smoothed IDF estimated on the training split, L2-normalised, and classified
with mini-batch-free sparse SGD.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from examples.lib.tokenize import tokenize


def _signed_hash(token: str, hash_size: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "little") % hash_size
    sign = 1.0 if digest[4] & 1 else -1.0
    return bucket, sign


@dataclass(frozen=True)
class SparseRow:
    indices: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.shape != self.values.shape:
            raise ValueError("indices and values must have the same length")


def hashed_counts(tokens: list[str], hash_size: int) -> dict[int, float]:
    acc: dict[int, float] = {}
    for token in tokens:
        bucket, sign = _signed_hash(token, hash_size)
        acc[bucket] = acc.get(bucket, 0.0) + sign
    return acc


def to_sparse(counts: dict[int, float], idf: np.ndarray | None = None) -> SparseRow:
    if not counts:
        return SparseRow(np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64))
    indices = np.fromiter(counts.keys(), dtype=np.int32)
    values = np.fromiter(counts.values(), dtype=np.float64)
    if idf is not None:
        values = values * idf[indices]
    norm = np.linalg.norm(values)
    if norm > 0:
        values = values / norm
    return SparseRow(indices, values)


def featurize_text(
    text: str,
    hash_size: int,
    idf: np.ndarray | None = None,
    drop_hashtags: bool = False,
) -> SparseRow:
    tokens = tokenize(text)
    if drop_hashtags:
        tokens = [token for token in tokens if not token.startswith("#")]
    return to_sparse(hashed_counts(tokens, hash_size), idf=idf)


def estimate_idf(count_rows: list[dict[int, float]], hash_size: int) -> np.ndarray:
    df = np.zeros(hash_size, dtype=np.float64)
    for counts in count_rows:
        if counts:
            df[list(counts.keys())] += 1.0
    n_docs = max(len(count_rows), 1)
    return np.log((n_docs + 1.0) / (df + 1.0)) + 1.0


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float
    hash_size: int
    idf: np.ndarray
    drop_hashtags: bool

    def score_row(self, row: SparseRow) -> float:
        if row.indices.size == 0:
            return self.bias
        return float(self.bias + np.dot(self.weights[row.indices], row.values))

    def predict_proba_text(self, text: str) -> float:
        row = featurize_text(
            text,
            self.hash_size,
            idf=self.idf,
            drop_hashtags=self.drop_hashtags,
        )
        return _sigmoid(self.score_row(row))

    def predict_text(self, text: str) -> int:
        return int(self.predict_proba_text(text) >= 0.5)


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -35.0, 35.0))
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(
    rows: list[SparseRow],
    labels: np.ndarray,
    hash_size: int,
    idf: np.ndarray,
    drop_hashtags: bool,
    epochs: int = 6,
    learning_rate: float = 0.15,
    l2: float = 1e-5,
    seed: int = 0,
) -> LogisticModel:
    rng = np.random.default_rng(seed)
    weights = np.zeros(hash_size, dtype=np.float64)
    bias = 0.0
    order = np.arange(len(rows))

    for epoch in range(epochs):
        rng.shuffle(order)
        lr = learning_rate / (1.0 + 0.2 * epoch)
        for i in order:
            row = rows[i]
            y = float(labels[i])
            z = bias if row.indices.size == 0 else bias + np.dot(weights[row.indices], row.values)
            err = _sigmoid(z) - y
            if row.indices.size:
                weights[row.indices] -= lr * (err * row.values + l2 * weights[row.indices])
            bias -= lr * err

    return LogisticModel(
        weights=weights,
        bias=bias,
        hash_size=hash_size,
        idf=idf,
        drop_hashtags=drop_hashtags,
    )


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_row(self) -> str:
        return (
            f"acc={self.accuracy:.3f}  f1={self.f1:.3f}  "
            f"p={self.precision:.3f}  r={self.recall:.3f}  "
            f"tp={self.tp} fp={self.fp} fn={self.fn} tn={self.tn}"
        )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return BinaryMetrics(acc, f1, precision, recall, tp, fp, fn, tn)


def evaluate(model: LogisticModel, texts: list[str], labels: np.ndarray) -> BinaryMetrics:
    preds = np.fromiter((model.predict_text(text) for text in texts), dtype=np.int64, count=len(texts))
    return binary_metrics(labels, preds)
