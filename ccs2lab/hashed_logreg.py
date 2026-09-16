"""Hashed-bag logistic regression with explicit cue features.

This is a classroom stand-in for the missing SVM / RF pickles. It is
not the 2023 model. The point is to measure how much of the official
test accuracy is sitting in ``#not`` / ``#sarcasm`` versus the rest of
the tweet.

Features:

* CRC32 feature hashing of tokens into ``hash_dim`` buckets
* L2-normalized bag-of-words
* concatenated binary cues from :mod:`ccs2lab.cues`

Training is mini-batch SGD with L2. Deterministic given ``seed``.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from ccs2lab.cues import CueProfile, profile_tokens
from ccs2lab.tokenize import tokenize_tweet

CUE_FEATURE_NAMES = (
    "not_tag",
    "sarcasm_tag",
    "explicit",
    "emoji",
    "contrast",
    "elongation",
    "question",
    "exclaim",
    "user",
    "positive",
)


def _hash_token(token: str, dim: int) -> int:
    return zlib.crc32(token.encode("utf-8")) % dim


def _cue_vector(profile: CueProfile) -> np.ndarray:
    flags = (
        profile.has_not_tag,
        profile.has_sarcasm_tag,
        profile.has_explicit,
        profile.has_emoji,
        profile.has_contrast,
        profile.has_elongation,
        profile.has_question,
        profile.has_exclaim,
        profile.has_user,
        profile.has_positive,
    )
    return np.asarray(flags, dtype=np.float64)


def featurize(
    text: str,
    *,
    hash_dim: int = 2048,
    use_cues: bool = True,
) -> np.ndarray:
    tokens = tokenize_tweet(text)
    bag = np.zeros((hash_dim,), dtype=np.float64)
    for token in tokens:
        bag[_hash_token(token, hash_dim)] += 1.0
    norm = np.linalg.norm(bag)
    if norm > 0:
        bag /= norm
    if not use_cues:
        return bag
    profile = profile_tokens(tokens)
    return np.concatenate([bag, _cue_vector(profile)])


def featurize_many(
    texts: list[str],
    *,
    hash_dim: int = 2048,
    use_cues: bool = True,
) -> np.ndarray:
    return np.stack(
        [featurize(text, hash_dim=hash_dim, use_cues=use_cues) for text in texts],
        axis=0,
    )


@dataclass
class LogisticModel:
    weights: np.ndarray
    bias: float
    hash_dim: int
    use_cues: bool

    def decision(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features[None, :]
        return features @ self.weights + self.bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.clip(self.decision(features), -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= 0.5).astype(np.int64)

    def predict_texts(self, texts: list[str]) -> np.ndarray:
        feats = featurize_many(texts, hash_dim=self.hash_dim, use_cues=self.use_cues)
        return self.predict(feats)

    def top_cue_weights(self) -> list[tuple[str, float]]:
        if not self.use_cues:
            return []
        cue_w = self.weights[self.hash_dim :]
        return sorted(
            zip(CUE_FEATURE_NAMES, (float(x) for x in cue_w)),
            key=lambda item: -abs(item[1]),
        )


def fit_logreg(
    texts: list[str],
    labels: list[int],
    *,
    hash_dim: int = 2048,
    use_cues: bool = True,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 0.35,
    l2: float = 1e-4,
    seed: int = 7,
) -> LogisticModel:
    rng = np.random.default_rng(seed)
    x = featurize_many(texts, hash_dim=hash_dim, use_cues=use_cues)
    y = np.asarray(labels, dtype=np.float64)
    n, dim = x.shape
    weights = np.zeros((dim,), dtype=np.float64)
    bias = 0.0
    order = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]
            logits = np.clip(xb @ weights + bias, -30, 30)
            pred = 1.0 / (1.0 + np.exp(-logits))
            err = pred - yb
            grad_w = (xb.T @ err) / len(idx) + l2 * weights
            grad_b = float(err.mean())
            weights -= lr * grad_w
            bias -= lr * grad_b
    return LogisticModel(weights=weights, bias=bias, hash_dim=hash_dim, use_cues=use_cues)
