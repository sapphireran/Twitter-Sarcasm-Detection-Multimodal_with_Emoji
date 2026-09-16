"""Sparse bag-of-words counts with a frozen vocabulary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class CountVectorizer:
    min_df: int = 2
    max_features: int = 20000
    add_bigrams: bool = False
    vocabulary_: dict[str, int] = field(default_factory=dict)
    document_frequency_: dict[str, int] = field(default_factory=dict)

    def _ngrams(self, tokens: list[str]) -> list[str]:
        grams = list(tokens)
        if self.add_bigrams:
            grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        return grams

    def fit(self, docs: list[list[str]]) -> CountVectorizer:
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(self._ngrams(tokens)))
        kept = [token for token, count in df.items() if count >= self.min_df]
        kept.sort(key=lambda token: (-df[token], token))
        if self.max_features is not None:
            kept = kept[: self.max_features]
        self.vocabulary_ = {token: idx for idx, token in enumerate(kept)}
        self.document_frequency_ = {token: df[token] for token in kept}
        return self

    def transform(self, docs: list[list[str]]) -> list[dict[int, int]]:
        if not self.vocabulary_:
            raise RuntimeError("CountVectorizer.fit() has not been called")
        rows: list[dict[int, int]] = []
        vocab = self.vocabulary_
        for tokens in docs:
            counts: dict[int, int] = {}
            for gram in self._ngrams(tokens):
                idx = vocab.get(gram)
                if idx is None:
                    continue
                counts[idx] = counts.get(idx, 0) + 1
            rows.append(counts)
        return rows

    def fit_transform(self, docs: list[list[str]]) -> list[dict[int, int]]:
        return self.fit(docs).transform(docs)

    @property
    def n_features(self) -> int:
        return len(self.vocabulary_)
