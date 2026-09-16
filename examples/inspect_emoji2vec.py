#!/usr/bin/env python3
"""Inspect the checked-in emoji2vec binaries without Gensim.

Prints vocab size, vector size, a few affect-bearing neighbours, and
(optionally) a comparison between emoji2vec_twitter.bin and emoji2vec.bin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EMOJI2VEC_ORIGINAL,
    EMOJI2VEC_TWITTER,
    cosine,
    load_word2vec_binary,
    nearest,
)

DEFAULT_QUERIES = ("😒", "😍", "😭", "😃", "🔫", "🌲", "😎", "💔")


def token_index(table: dict) -> dict:
    return {tok: i for i, tok in enumerate(table["tokens"])}


def show_table(table: dict, queries, k: int) -> None:
    print(f"file:         {table['path']}")
    print(f"header vocab: {table['vocab_size']}")
    print(f"loaded rows:  {len(table['tokens'])}")
    print(f"vector size:  {table['vector_size']}")
    norms = (table["vectors"] ** 2).sum(axis=1) ** 0.5
    print(
        f"L2 norms:     min={float(norms.min()):.4f}  "
        f"mean={float(norms.mean()):.4f}  max={float(norms.max()):.4f}"
    )
    print(f"first tokens: {table['tokens'][:12]}")
    index = token_index(table)
    print()
    for query in queries:
        if query not in index:
            print(f"  {query!r}: not in vocab")
            continue
        q_i = index[query]
        vec = table["vectors"][q_i]
        neighbours = nearest(vec, table["vectors"], k=k + 1)
        # drop self
        neighbours = [j for j in neighbours if j != q_i][:k]
        parts = []
        for j in neighbours:
            parts.append(
                f"{table['tokens'][j]}={cosine(vec, table['vectors'][j]):.3f}"
            )
        print(
            f"  {query}  norm={float(norms[q_i]):.3f}  "
            f"neighbors: {', '.join(parts)}"
        )
    print()


def compare(twitter: dict, original: dict, queries) -> None:
    t_index = token_index(twitter)
    o_index = token_index(original)
    shared = sorted(set(t_index) & set(o_index))
    print(f"shared vocab: {len(shared)} / twitter={len(t_index)} original={len(o_index)}")
    print(
        f"dims: twitter={twitter['vector_size']}  original={original['vector_size']}"
    )
    if twitter["vector_size"] != original["vector_size"]:
        print(
            "cosine(twitter, original) is undefined: the Twitter table is "
            "aligned to 200-d GloVe-Twitter, the upstream dump is 300-d "
            "(Google-News word2vec space). Neighbour lists above are the "
            "right comparison, not a cross-space cosine."
        )
        print("tokens present in only one file:")
        only_t = sorted(set(t_index) - set(o_index))
        only_o = sorted(set(o_index) - set(t_index))
        print(f"  twitter-only ({len(only_t)}): {only_t[:20]}")
        print(f"  original-only ({len(only_o)}): {only_o[:20]}")
        print()
        return
    sims = []
    for tok in shared:
        sims.append(
            cosine(
                twitter["vectors"][t_index[tok]],
                original["vectors"][o_index[tok]],
            )
        )
    import numpy as np

    arr = np.asarray(sims, dtype=float)
    print(
        f"cosine(twitter, original) on shared keys: "
        f"min={arr.min():.3f}  mean={arr.mean():.3f}  "
        f"median={float(np.median(arr)):.3f}  max={arr.max():.3f}"
    )
    print("per-query cosine between the two files:")
    for query in queries:
        if query in t_index and query in o_index:
            s = cosine(
                twitter["vectors"][t_index[query]],
                original["vectors"][o_index[query]],
            )
            print(f"  {query}  {s:.3f}")
        else:
            print(f"  {query}  missing in one of the files")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        choices=("twitter", "original", "both"),
        default="twitter",
        help="Which binary to inspect. Default: the table the notebooks load.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also report cosine(twitter, original) on the shared vocab.",
    )
    parser.add_argument(
        "--neighbors",
        nargs="*",
        default=None,
        help="Emoji / tokens to look up. Default: a small affect set.",
    )
    parser.add_argument("--k", type=int, default=6, help="Neighbour count.")
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Load only the first N rows (debug).",
    )
    args = parser.parse_args(argv)
    queries = tuple(args.neighbors) if args.neighbors else DEFAULT_QUERIES

    want_twitter = args.file in ("twitter", "both") or args.compare
    want_original = args.file in ("original", "both") or args.compare

    twitter = (
        load_word2vec_binary(EMOJI2VEC_TWITTER, max_words=args.max_words)
        if want_twitter
        else None
    )
    original = (
        load_word2vec_binary(EMOJI2VEC_ORIGINAL, max_words=args.max_words)
        if want_original
        else None
    )

    if args.file in ("twitter", "both") and twitter is not None:
        show_table(twitter, queries, args.k)
    if args.file in ("original", "both") and original is not None:
        show_table(original, queries, args.k)
    if args.compare and twitter is not None and original is not None:
        compare(twitter, original, queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
