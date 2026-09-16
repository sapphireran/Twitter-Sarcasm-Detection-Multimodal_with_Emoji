#!/usr/bin/env python3
"""Show how the sklearn stack builds tweet vectors (mean pooling).

``data_utils.AverageVectorPerTweet`` averages every in-vocab token and
writes a zero vector when nothing hits. This script does the same thing
with a *synthetic* 8-d table so you can see the arithmetic on real tweets
without downloading GloVe.

It also mimics the WE concatenation: word-mean ‖ emoji-mean.

Run::

    python3 examples/07_average_pooling.py
    python3 examples/07_average_pooling.py --n 5 --split subtest
    python3 examples/07_average_pooling.py --real-emoji2vec --n 3
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.emoji import emoji_code_points
from examples.common.io import load_split
from examples.common.tokenize import tokenize_tweet
from examples.common.word2vec import load_word2vec_binary, lookup_emoji


def hashed_vector(token: str, dim: int, seed: int) -> np.ndarray:
    """Stable pseudo-embedding so demos do not need GloVe."""
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vec = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n


def mean_pool(tokens, lookup, dim: int) -> tuple[np.ndarray, int]:
    rows = []
    for tok in tokens:
        vec = lookup(tok)
        if vec is not None:
            rows.append(vec)
    if not rows:
        return np.zeros((dim,), dtype=np.float32), 0
    return np.mean(np.stack(rows, axis=0), axis=0), len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="subtest", choices=["train", "test", "subtest"])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--real-emoji2vec",
        action="store_true",
        help="use emoji2vec_twitter.bin for the emoji half (200-d)",
    )
    args = parser.parse_args()

    split = load_split(args.split)
    emoji_table = None
    emoji_dim = args.dim
    if args.real_emoji2vec:
        emoji_table = load_word2vec_binary(ROOT / "emoji2vec_twitter.bin")
        emoji_dim = emoji_table.dim

    print(f"split={split.name}  showing {args.n} tweets")
    print("word vectors are hashed unit vectors (stand-in for GloVe).")
    if emoji_table is None:
        print("emoji vectors are hashed too; pass --real-emoji2vec to use the bin.")
    else:
        print(f"emoji vectors from emoji2vec_twitter.bin (dim={emoji_dim}).")
    print()

    shown = 0
    for text, label in split.pairs():
        if shown >= args.n:
            break
        tokens = tokenize_tweet(text)
        emojis = emoji_code_points(text)

        word_mean, n_word = mean_pool(
            tokens,
            lambda t: hashed_vector(t, args.dim, args.seed),
            args.dim,
        )

        if emoji_table is not None:
            emo_mean, n_emo = mean_pool(
                emojis,
                lambda t, table=emoji_table: lookup_emoji(table, t),
                emoji_dim,
            )
        else:
            emo_mean, n_emo = mean_pool(
                emojis,
                lambda t: hashed_vector("E:" + t, args.dim, args.seed + 1),
                args.dim,
            )

        we = np.concatenate([word_mean, emo_mean])
        tag = "sarcastic" if label == 1 else "literal"
        print(
            f"[{shown+1}] {tag}  tokens={len(tokens)}  "
            f"word_hits={n_word}  emoji_hits={n_emo}"
        )
        print(f"    {text[:120]}")
        print(
            "    word_mean[:4]  = "
            + np.array2string(word_mean[:4], precision=3, suppress_small=True)
        )
        print(
            "    emoji_mean[:4] = "
            + np.array2string(emo_mean[:4], precision=3, suppress_small=True)
        )
        print(f"    WE dim = {we.shape[0]}  (word {args.dim} + emoji {emoji_dim})")
        if n_emo == 0:
            print("    emoji_mean is zeros — WE adds no information on this row.")
        print()
        shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
