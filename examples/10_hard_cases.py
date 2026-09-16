#!/usr/bin/env python3
"""List sarcastic test tweets that have no cue hashtag and no emoji.

Those are the rows where mean-pooled emoji2vec cannot help and a
hashtag rule predicts 'not sarcastic'. The 2023 BiLSTM still has the
word order; this script only surfaces the text so a write-up can quote
real misses of the cheap baselines.
"""

from __future__ import annotations

import argparse
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
    strip_wrapping_quotes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "test", "subtest"))
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)

    root = repo_root()
    sent_path = root / "dataset" / f"{args.split}_sentence.csv"
    raw_lines = sent_path.read_text(encoding="utf-8", errors="replace").splitlines()
    docs, labels = read_sentence_label_pair(sent_path, root / "dataset" / f"{args.split}_label.csv")

    hard = []
    easy_cue = 0
    emoji_only = 0
    for raw, doc, y in zip(raw_lines, docs, labels):
        if int(y) != 1:
            continue
        cue = has_cue_hashtag(doc)
        emo = any(is_emoji_token(t) for t in doc)
        if cue:
            easy_cue += 1
        elif emo:
            emoji_only += 1
        else:
            hard.append(strip_wrapping_quotes(raw))

    print(f"split={args.split} sarcastic total={int(labels.sum())}")
    print(f"  with cue hashtag:              {easy_cue}")
    print(f"  no cue, but has emoji:         {emoji_only}")
    print(f"  no cue, no emoji (hard):       {len(hard)}")
    print()
    print(f"first {min(args.limit, len(hard))} hard sarcastic tweets:")
    for i, text in enumerate(hard[: args.limit], start=1):
        print(f"{i:2d}. {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
