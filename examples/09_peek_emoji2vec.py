#!/usr/bin/env python3
"""Read ``emoji2vec_twitter.bin`` with the stdlib + NumPy, no gensim.

Confirms the table the 2023 notebooks loaded: 200-d vectors, one row per
emoji (and a few regional-indicator flags). Prints coverage on the test
split: how often ``AverageVectorPerEmoji`` would have found a hit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (  # noqa: E402
    is_emoji_token,
    iter_word2vec_binary,
    read_sentence_label_pair,
    repo_root,
)


def main() -> int:
    path = repo_root() / "emoji2vec_twitter.bin"
    if not path.exists():
        print("missing emoji2vec_twitter.bin", file=sys.stderr)
        return 1

    keys = []
    dim = None
    norms = []
    for token, vec in iter_word2vec_binary(path):
        keys.append(token)
        dim = int(vec.shape[0])
        norms.append(float(np.linalg.norm(vec)))

    print(f"file          {path.name}")
    print(f"rows          {len(keys)}")
    print(f"dim           {dim}")
    print(f"norm mean/min/max  {np.mean(norms):.3f} / {min(norms):.3f} / {max(norms):.3f}")
    print(f"first eight   {keys[:8]}")
    # A few common face emoji the fixture uses.
    wanted = ["😒", "😭", "😅", "😃", "😑", "😌", "👏"]
    table = {k: None for k in keys}
    print("fixture faces in table:")
    for e in wanted:
        print(f"  {e}  {'yes' if e in table else 'NO'}")

    docs, labels = read_sentence_label_pair(
        repo_root() / "dataset" / "test_sentence.csv",
        repo_root() / "dataset" / "test_label.csv",
    )
    tweets_with_emoji = 0
    tweets_with_hit = 0
    oov = 0
    hits = 0
    for doc in docs:
        emojis = [t for t in doc if is_emoji_token(t)]
        if not emojis:
            continue
        tweets_with_emoji += 1
        this_hits = [t for t in emojis if t in table]
        hits += len(this_hits)
        oov += len(emojis) - len(this_hits)
        if this_hits:
            tweets_with_hit += 1
    print()
    print(f"test tweets with an emoji token:     {tweets_with_emoji}")
    print(f"those with ≥1 emoji2vec hit:         {tweets_with_hit}")
    print(f"emoji token hits / OOV on test:      {hits} / {oov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
