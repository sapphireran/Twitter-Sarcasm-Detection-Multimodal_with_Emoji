#!/usr/bin/env python3
"""Read the checked-in emoji2vec binaries and print nearest neighbors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccs2lab.paths import EMOJI2VEC_300, EMOJI2VEC_TWITTER_200
from ccs2lab.tokenize import tokenize_tweet
from ccs2lab.w2v import average_present, load_word2vec_bin, read_word2vec_header


DEFAULT_PROBES = ("😂", "😒", "😭", "❤", "🔫")


def _dump(path: Path, probes: list[str], k: int) -> None:
    n, dim = read_word2vec_header(path)
    print(f"## {path.name}")
    print(f"header: {n} vectors × {dim} dim, bytes={path.stat().st_size}")
    table = load_word2vec_bin(path)
    if len(table) != n or table.dim != dim:
        raise RuntimeError("header and body disagree")
    print(f"loaded: {len(table)} × {table.dim}")
    for emoji in probes:
        if emoji not in table:
            print(f"  {emoji}: OOV")
            continue
        neighbors = table.most_similar(emoji, k=k)
        pretty = ", ".join(f"{tok} {score:.3f}" for tok, score in neighbors)
        print(f"  {emoji} → {pretty}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--probes",
        default=",".join(DEFAULT_PROBES),
        help="Comma-separated emoji to look up.",
    )
    parser.add_argument(
        "--tweet",
        default="I loovee when people text back ... 😒 #sarcastictweet",
        help="Optional tweet for a mean-pool demo on the 200d table.",
    )
    args = parser.parse_args()
    probes = [item for item in args.probes.split(",") if item]
    _dump(EMOJI2VEC_TWITTER_200, probes, args.k)
    _dump(EMOJI2VEC_300, probes, args.k)

    table = load_word2vec_bin(EMOJI2VEC_TWITTER_200)
    tokens = tokenize_tweet(args.tweet)
    vec = average_present(tokens, table)
    hits = [tok for tok in tokens if tok in table]
    print("## mean-pool demo (200d twitter table)")
    print(f"tweet: {args.tweet}")
    print(f"tokens: {tokens}")
    print(f"in-vocab emoji: {hits}")
    print(f"mean L2: {float((vec ** 2).sum() ** 0.5):.4f}")
    if hits:
        near = table.nearest(vec, k=5)
        print("nearest to mean:", ", ".join(f"{t} {s:.3f}" for t, s in near))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
