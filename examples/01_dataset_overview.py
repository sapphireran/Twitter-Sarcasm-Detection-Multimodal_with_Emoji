#!/usr/bin/env python3
"""Print split sizes, emoji rates, and cue-hashtag rates from dataset/*.csv.

This is the script behind the tables in docs/dataset.md. Re-run it after any
CSV edit; the docs should change with the output, not the other way around.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (  # noqa: E402
    has_cue_hashtag,
    is_emoji_token,
    read_sentence_label_pair,
    repo_root,
)


def summarize(split: str) -> dict:
    root = repo_root()
    docs, labels = read_sentence_label_pair(
        root / "dataset" / f"{split}_sentence.csv",
        root / "dataset" / f"{split}_label.csv",
    )
    n = len(labels)
    pos = int(labels.sum())
    emoji_pos = emoji_neg = cue_pos = cue_neg = 0
    lengths = []
    hashes: collections.Counter[str] = collections.Counter()
    for doc, y in zip(docs, labels):
        lengths.append(len(doc))
        has_emoji = any(is_emoji_token(t) for t in doc)
        has_cue = has_cue_hashtag(doc)
        if has_emoji:
            if y:
                emoji_pos += 1
            else:
                emoji_neg += 1
        if has_cue:
            if y:
                cue_pos += 1
            else:
                cue_neg += 1
        for tok in doc:
            if tok.startswith("#"):
                hashes[tok] += 1
    lengths_sorted = sorted(lengths)
    return {
        "split": split,
        "n": n,
        "pos": pos,
        "neg": n - pos,
        "emoji": emoji_pos + emoji_neg,
        "emoji_pos": emoji_pos,
        "emoji_neg": emoji_neg,
        "cue": cue_pos + cue_neg,
        "cue_pos": cue_pos,
        "cue_neg": cue_neg,
        "len_min": lengths_sorted[0],
        "len_med": lengths_sorted[n // 2],
        "len_max": lengths_sorted[-1],
        "top_hashtags": hashes.most_common(8),
    }


def main() -> int:
    print("split    n     pos    neg   emoji  emoji|1 emoji|0   cue  cue|1 cue|0  min/med/max tok")
    print("-" * 96)
    for split in ("train", "test", "subtest"):
        s = summarize(split)
        print(
            f"{s['split']:<8} {s['n']:5d} {s['pos']:6d} {s['neg']:6d} "
            f"{s['emoji']:6d} {s['emoji_pos']:7d} {s['emoji_neg']:7d} "
            f"{s['cue']:5d} {s['cue_pos']:5d} {s['cue_neg']:5d}  "
            f"{s['len_min']}/{s['len_med']}/{s['len_max']}"
        )
    print()
    for split in ("train", "test", "subtest"):
        s = summarize(split)
        tops = ", ".join(f"{h}={c}" for h, c in s["top_hashtags"])
        print(f"{split} top hashtags: {tops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
