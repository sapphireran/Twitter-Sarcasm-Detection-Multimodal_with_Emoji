"""Build the shared 200-d Keras-style table with and without emoji fallback."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .fusion import build_shared_embedding_matrix, encode_sequence, toy_tables
from .tokenize import tokenize_corpus


DEFAULT_DOCS = (
    "I love monday mornings #not 😒",
    "Great day 😃",
    "I hate walking to school",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4)
    return parser


def _nonzero_rows(matrix: np.ndarray) -> int:
    return int(np.count_nonzero(np.linalg.norm(matrix, axis=1)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    word_table, emoji_table = toy_tables(dim=args.dim)
    docs = tokenize_corpus(DEFAULT_DOCS)
    pad = max(len(doc) for doc in docs)

    print("Neural path: one matrix. Emoji either stay zero or copy emoji2vec.")
    print("This is different from concatenating a second 200-d channel.")
    print()

    for use_emoji in (False, True):
        index, matrix, fallbacks = build_shared_embedding_matrix(
            docs, word_table, emoji_table, args.dim, use_emoji_fallback=use_emoji
        )
        mode = "word+emoji fallback" if use_emoji else "word only"
        print(f"=== {mode} ===")
        print(f"vocab {len(index)}  matrix {matrix.shape}  nonzero rows {_nonzero_rows(matrix)}")
        print(f"emoji fallbacks filled: {fallbacks or 'none'}")
        encoded = encode_sequence(DEFAULT_DOCS[0], index, matrix, pad_to=pad)
        print(f"example ids: {encoded.ids}")
        print(f"row L2 norms: {np.round(np.linalg.norm(encoded.matrix, axis=1), 3)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
