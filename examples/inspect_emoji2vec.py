#!/usr/bin/env python3
"""Inspect the shipped emoji2vec table without downloading GloVe.

Requires gensim. The file emoji2vec_twitter.bin already lives in the repo
root; this script only reads it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarcasm_lib.paths import EMOJI2VEC_TWITTER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neighbors",
        default="😒",
        help="emoji (or token) whose nearest neighbours will be printed",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--path",
        type=Path,
        default=EMOJI2VEC_TWITTER,
        help="KeyedVectors binary to load",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:
        raise SystemExit(
            "gensim is not installed. pip install gensim  "
            "or skip this script; the other examples do not need it."
        ) from exc

    if not args.path.exists():
        raise SystemExit(f"missing emoji table: {args.path}")

    print(f"loading {args.path} ...")
    model = KeyedVectors.load_word2vec_format(str(args.path), binary=True)
    vocab = getattr(model, "index_to_key", None) or list(model.key_to_index)
    print(f"tokens: {len(vocab)}  dim: {model.vector_size}")

    query = args.neighbors
    in_vocab = query in model.key_to_index if hasattr(model, "key_to_index") else query in model
    if not in_vocab:
        sample = " ".join(vocab[:20])
        raise SystemExit(
            f"{query!r} is not in the table. First tokens: {sample}"
        )

    print(f"\nnearest to {query}:")
    for token, score in model.most_similar(query, topn=args.k):
        print(f"  {score:6.3f}  {token}")


if __name__ == "__main__":
    main()
