"""Read a word2vec binary file without Gensim.

The on-disk layout matches what ``KeyedVectors.load_word2vec_format``
expects for ``binary=True``: a text header ``vocab_size dim\\n``, then
for each row a UTF-8 token, a space, and ``dim`` little-endian float32s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Word2VecTable:
    tokens: tuple[str, ...]
    vectors: np.ndarray  # (V, D), float32
    index: dict[str, int]

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.index

    def __getitem__(self, token: str) -> np.ndarray:
        try:
            return self.vectors[self.index[token]]
        except KeyError as exc:
            raise KeyError(token) from exc


def load_word2vec_binary(path: Path | str) -> Word2VecTable:
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.readline()
        try:
            vocab_size_s, dim_s = header.split()
            vocab_size, dim = int(vocab_size_s), int(dim_s)
        except ValueError as exc:
            raise ValueError(f"{path}: bad word2vec header {header!r}") from exc

        row_bytes = dim * 4
        tokens: list[str] = []
        matrix = np.empty((vocab_size, dim), dtype=np.float32)

        for row in range(vocab_size):
            raw_token = bytearray()
            while True:
                ch = handle.read(1)
                if not ch:
                    raise EOFError(f"{path}: ended inside token {row}")
                if ch == b" ":
                    break
                if ch != b"\n":
                    raw_token.extend(ch)
            token = raw_token.decode("utf-8", errors="replace")
            payload = handle.read(row_bytes)
            if len(payload) != row_bytes:
                raise EOFError(f"{path}: short vector for {token!r}")
            matrix[row] = np.frombuffer(payload, dtype="<f4")
            tokens.append(token)

    index = {token: i for i, token in enumerate(tokens)}
    return Word2VecTable(tokens=tuple(tokens), vectors=matrix, index=index)


def cosine_neighbours(
    table: Word2VecTable,
    token: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return the ``k`` nearest *other* rows by cosine similarity."""
    query = table[token]
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []
    norms = np.linalg.norm(table.vectors, axis=1)
    dots = table.vectors @ query
    sims = np.divide(dots, norms * query_norm, out=np.zeros(len(table)), where=norms > 0)
    query_i = table.index[token]
    sims[query_i] = -np.inf
    top = np.argpartition(-sims, min(k, len(table) - 1))[:k]
    ranked = sorted(((table.tokens[i], float(sims[i])) for i in top), key=lambda kv: -kv[1])
    return ranked[:k]


def mean_in_vocab(table: Word2VecTable, tokens: list[str]) -> np.ndarray | None:
    rows = [table[token] for token in tokens if token in table]
    if not rows:
        return None
    return np.mean(np.stack(rows, axis=0), axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def header_only(path: Path | str) -> tuple[int, int]:
    """Parse just ``vocab_size, dim`` from the first line."""
    with Path(path).open("rb") as handle:
        parts = handle.readline().split()
    if len(parts) != 2:
        raise ValueError(f"{path}: expected two header integers")
    return int(parts[0]), int(parts[1])
