"""Read Google word2vec binary files without gensim.

The archive ships two emoji tables:

* ``emoji2vec.bin`` — 1,661 × 300, Eisner et al. original space
* ``emoji2vec_twitter.bin`` — 1,661 × 200, the table the 2023
  notebooks concatenated onto GloVe Twitter 27B 200d

Both headers are ``vocab_size dim\\n`` followed by
``token[space]float32*dim`` records. Tokens may be multi-byte emoji.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ccs2lab.paths import EMOJI2VEC_300, EMOJI2VEC_TWITTER_200


@dataclass(frozen=True)
class KeyedVectors:
    tokens: tuple[str, ...]
    matrix: np.ndarray  # (n, dim), float32
    index: dict[str, int]

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self.index

    def __getitem__(self, token: str) -> np.ndarray:
        try:
            return self.matrix[self.index[token]]
        except KeyError as exc:
            raise KeyError(token) from exc

    def get(self, token: str, default: np.ndarray | None = None) -> np.ndarray | None:
        idx = self.index.get(token)
        if idx is None:
            return default
        return self.matrix[idx]

    def most_similar(self, token: str, k: int = 8) -> list[tuple[str, float]]:
        if token not in self.index:
            raise KeyError(token)
        vec = self.matrix[self.index[token]]
        return self.nearest(vec, k=k, skip={token})

    def nearest(self, vec: np.ndarray, k: int = 8, skip: set[str] | None = None) -> list[tuple[str, float]]:
        skip = skip or set()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        table_norm = np.linalg.norm(self.matrix, axis=1)
        table_norm = np.where(table_norm == 0, 1.0, table_norm)
        scores = (self.matrix @ vec) / (table_norm * norm)
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for idx in order:
            token = self.tokens[int(idx)]
            if token in skip:
                continue
            out.append((token, float(scores[int(idx)])))
            if len(out) >= k:
                break
        return out


def read_word2vec_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.readline()
    parts = header.split()
    if len(parts) != 2:
        raise ValueError(f"{path}: expected 'n dim' header, got {header!r}")
    return int(parts[0]), int(parts[1])


def load_word2vec_bin(path: Path) -> KeyedVectors:
    with path.open("rb") as handle:
        header = handle.readline()
        parts = header.split()
        if len(parts) != 2:
            raise ValueError(f"{path}: expected 'n dim' header, got {header!r}")
        count, dim = int(parts[0]), int(parts[1])
        tokens: list[str] = []
        matrix = np.empty((count, dim), dtype=np.float32)
        for row in range(count):
            raw = bytearray()
            while True:
                ch = handle.read(1)
                if not ch:
                    raise EOFError(f"{path}: truncated token at row {row}")
                if ch == b" ":
                    break
                raw.extend(ch)
            blob = handle.read(4 * dim)
            if len(blob) != 4 * dim:
                raise EOFError(f"{path}: truncated vector at row {row}")
            tokens.append(raw.decode("utf-8", errors="replace"))
            matrix[row] = np.frombuffer(blob, dtype="<f4")
    index = {token: i for i, token in enumerate(tokens)}
    return KeyedVectors(tokens=tuple(tokens), matrix=matrix, index=index)


def load_emoji2vec(*, twitter: bool = True) -> KeyedVectors:
    path = EMOJI2VEC_TWITTER_200 if twitter else EMOJI2VEC_300
    return load_word2vec_bin(path)


def average_present(tokens: list[str], table: KeyedVectors, *, dim: int | None = None) -> np.ndarray:
    """Mean of in-vocabulary tokens, else a zero vector.

    Matches ``AverageVectorPerTweet`` / ``AverageVectorPerEmoji`` in
    ``data_utils.py``: skip OOV, and if nothing hit, return zeros.
    """
    width = dim if dim is not None else table.dim
    rows = [table[token] for token in tokens if token in table]
    if not rows:
        return np.zeros((width,), dtype=np.float32)
    return np.mean(np.stack(rows, axis=0), axis=0)
