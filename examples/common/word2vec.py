"""Minimal reader for the word2vec / emoji2vec binary format.

``emoji2vec_twitter.bin`` in this repo is a 1,661 × 200 matrix in the original
Google word2vec binary layout (one text header line, then ``token<space>`` +
``dim`` little-endian float32 values). That is the same format Gensim's
``KeyedVectors.load_word2vec_format(..., binary=True)`` consumes, which is what
the 2023 notebooks used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class KeyedVectors:
    """Tiny stand-in for ``gensim.models.KeyedVectors``."""

    index: Dict[str, np.ndarray]
    dim: int

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def __getitem__(self, key: str) -> np.ndarray:
        return self.index[key]

    def __len__(self) -> int:
        return len(self.index)

    @property
    def vocab(self) -> Dict[str, np.ndarray]:
        """Gensim 3.x compatibility: ``if token in model.vocab``."""
        return self.index

    def keys(self) -> Iterable[str]:
        return self.index.keys()

    def vector_matrix(self) -> Tuple[List[str], np.ndarray]:
        keys = list(self.index.keys())
        matrix = np.stack([self.index[k] for k in keys], axis=0)
        return keys, matrix


def load_word2vec_binary(path: Path | str) -> KeyedVectors:
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.readline()
        try:
            vocab_size, dim = map(int, header.decode("utf-8").split())
        except ValueError as exc:
            raise ValueError(f"{path} is not a word2vec binary file") from exc
        index: Dict[str, np.ndarray] = {}
        for _ in range(vocab_size):
            chars: List[bytes] = []
            while True:
                ch = handle.read(1)
                if not ch:
                    break
                if ch == b" ":
                    break
                if ch != b"\n":
                    chars.append(ch)
            token = b"".join(chars).decode("utf-8", errors="replace")
            raw = handle.read(dim * 4)
            if len(raw) != dim * 4:
                raise ValueError(f"{path}: truncated vector for {token!r}")
            index[token] = np.frombuffer(raw, dtype=np.float32).copy()
    return KeyedVectors(index=index, dim=dim)


VS16 = "\uFE0F"


def lookup_emoji(vectors: KeyedVectors, token: str) -> np.ndarray | None:
    """Resolve a pictograph against emoji2vec, trying variation-selector variants.

    This dataset often stores U+2764 HEART (``❤``) while emoji2vec stores
    U+2764 + U+FE0F (``❤️``). The 2023 ``emoji`` package usually emits the
    fully-qualified sequence; a raw code-point walk does not.
    """
    candidates = [token, token + VS16, token.replace(VS16, "")]
    # Also try each code point of a ZWJ sequence independently.
    if "\u200d" in token:
        candidates.extend(token.split("\u200d"))
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if cand in vectors:
            return vectors[cand]
    return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def nearest(
    query: np.ndarray,
    vectors: KeyedVectors,
    k: int = 8,
    exclude: Sequence[str] = (),
) -> List[Tuple[str, float]]:
    """Return the ``k`` tokens with highest cosine similarity to ``query``."""
    skip = set(exclude)
    scored: List[Tuple[str, float]] = []
    for token, vec in vectors.index.items():
        if token in skip:
            continue
        scored.append((token, cosine(query, vec)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
