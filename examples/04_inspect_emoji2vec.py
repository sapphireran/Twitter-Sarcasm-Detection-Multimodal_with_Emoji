#!/usr/bin/env python3
"""Inspect the bundled emoji2vec tables without Gensim.

``emoji2vec_twitter.bin`` is 1,661 × 200 (aligned with GloVe-Twitter 200d).
``emoji2vec.bin`` is the original 1,661 × 300 table from Eisner et al. 2016.

This script:

1. loads both files with ``examples.common.word2vec``
2. prints nearest neighbours for a handful of pictographs
3. reports which dataset emojis are missing from the 200-d table

Run::

    python3 examples/04_inspect_emoji2vec.py
    python3 examples/04_inspect_emoji2vec.py --query 😒 😑 ❤ 😂
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.emoji import emoji_code_points
from examples.common.io import load_split
from examples.common.word2vec import cosine, load_word2vec_binary, lookup_emoji, nearest


DEFAULT_QUERIES = ["😂", "😒", "😑", "❤", "❤️", "😍", "😭", "😊", "😡"]


def coverage(vectors, split_name: str = "train") -> dict:
    split = load_split(split_name)
    present = Counter()
    missing = Counter()
    for text in split.texts:
        for e in set(emoji_code_points(text)):
            if lookup_emoji(vectors, e) is not None:
                present[e] += 1
            else:
                missing[e] += 1
    return {
        "n_types_present": len(present),
        "n_types_missing": len(missing),
        "n_tokens_present": sum(present.values()),
        "n_tokens_missing": sum(missing.values()),
        "top_missing": missing.most_common(12),
        "top_present": present.most_common(12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", nargs="*", default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    twitter = load_word2vec_binary(ROOT / "emoji2vec_twitter.bin")
    original = load_word2vec_binary(ROOT / "emoji2vec.bin")
    print(f"emoji2vec_twitter.bin  vocab={len(twitter)}  dim={twitter.dim}")
    print(f"emoji2vec.bin          vocab={len(original)}  dim={original.dim}")
    print()

    print("Nearest neighbours in the 200-d Twitter-aligned table")
    print("----------------------------------------------------")
    for q in args.query:
        vec = lookup_emoji(twitter, q)
        if vec is None:
            print(f"  {q}  not in vocab (tried FE0F variants)")
            continue
        nbrs = nearest(vec, twitter, k=args.k, exclude=[q, q + "\ufe0f"])
        nbr_str = ", ".join(f"{tok} ({sim:.3f})" for tok, sim in nbrs)
        note = "" if q in twitter else "  [resolved via U+FE0F]"
        print(f"  {q}  ->  {nbr_str}{note}")
    print()

    heart = lookup_emoji(twitter, "❤")
    if "😒" in twitter and heart is not None and "😑" in twitter:
        print(
            f"cosine(😒, ❤/❤️) = {cosine(twitter['😒'], heart):.3f}   "
            "(deadpan vs. affectionate — should be far apart)"
        )
        print(
            f"cosine(😒, 😑)     = {cosine(twitter['😒'], twitter['😑']):.3f}   "
            "(both are flat-affect faces)"
        )
        print()

    cov = coverage(twitter, "train")
    print("Train-split coverage by emoji2vec_twitter.bin (with FE0F fallback)")
    print("----------------------------------------------------------------")
    print(
        f"  types present={cov['n_types_present']}  types missing={cov['n_types_missing']}"
    )
    print(
        f"  token df present={cov['n_tokens_present']}  "
        f"token df missing={cov['n_tokens_missing']}"
    )
    print("  most common OOV emoji types:")
    for e, n in cov["top_missing"]:
        print(f"    {e}  df={n}")
    print()
    print("The original preprocessor looks up tokens in GloVe first, then tries")
    print("emoji.emoji_list on the leftover string. Combined emoji (ZWJ families,")
    print("skin tones) often miss the table and become the 200-d zero vector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
