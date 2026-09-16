#!/usr/bin/env python3
"""Mean-pool a 16-d toy GloVe / emoji2vec pair the way ``ml_read_data`` does.

A real GloVe-Twitter 200d table is hundreds of megabytes. The fusion
*shape* does not need it: words look up one table, emoji look up the
other, empty sides become zeros, multimodal is the concatenation.

This script builds two tiny hash-based tables, pools the 12-tweet
fixture, and prints which tweets actually used the emoji half.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (  # noqa: E402
    build_embed_table,
    docs_to_padded,
    is_emoji_token,
    pooled_views,
    read_sentence_label_pair,
)

WIDTH = 16
FIXTURE_S = Path(__file__).resolve().parent / "fixtures" / "tiny_sentence.csv"
FIXTURE_L = Path(__file__).resolve().parent / "fixtures" / "tiny_label.csv"


def hash_vec(token: str, salt: bytes, width: int) -> np.ndarray:
    """Deterministic unit-ish vector so the script needs no seed file."""
    digest = hashlib.sha256(salt + token.encode("utf-8")).digest()
    # 32 bytes → repeat to width, map to [-1, 1]
    raw = np.frombuffer((digest * ((width // 8) + 1))[:width], dtype=np.uint8)
    vec = (raw.astype(np.float64) / 127.5) - 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def tables_for(docs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    words: dict[str, np.ndarray] = {}
    emoji: dict[str, np.ndarray] = {}
    for doc in docs:
        for tok in doc:
            if is_emoji_token(tok):
                emoji[tok] = hash_vec(tok, b"emoji2vec-toy", WIDTH)
            else:
                words[tok] = hash_vec(tok, b"glove-toy", WIDTH)
    return words, emoji


def main() -> int:
    docs, labels = read_sentence_label_pair(FIXTURE_S, FIXTURE_L)
    word_table, emoji_table = tables_for(docs)
    x_word, x_multi = pooled_views(
        docs, word_table, emoji_table, word_width=WIDTH, emoji_width=WIDTH
    )
    print(f"word view   {x_word.shape}   (N, {WIDTH})")
    print(f"multi view  {x_multi.shape}   (N, {WIDTH * 2})")
    print()
    print("tweet                              y  |emoji|_2  used emoji rows")
    print("-" * 72)
    for doc, y, multi in zip(docs, labels, x_multi):
        emoji_half = multi[WIDTH:]
        used = [t for t in doc if t in emoji_table]
        text = " ".join(doc)[:32]
        print(
            f"{text:<32}  {int(y)}  {np.linalg.norm(emoji_half):8.3f}  {used or '—'}"
        )

    vocab, matrix = build_embed_table(docs, word_table, emoji_table, width=WIDTH)
    padded = docs_to_padded(docs, vocab)
    print()
    print(f"sequence vocab {len(vocab)}  matrix {matrix.shape}  padded {padded.shape}")
    print("index 0 is pad (all zeros):", bool(np.allclose(matrix[0], 0)))
    # Count how many rows came from the emoji table.
    emoji_rows = 0
    for tok, idx in vocab.items():
        if is_emoji_token(tok) and tok in emoji_table:
            emoji_rows += 1
    print(f"emoji token rows written into the matrix: {emoji_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
