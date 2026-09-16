#!/usr/bin/env python3
"""Show that subtest is almost the emoji-bearing slice of test.

docs/dataset.md claims the emoji-positive counts match (164 sarcastic,
102 not) and that subtest adds a few extra rows. This script checks
that claim by set membership on stripped sentence text.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (  # noqa: E402
    is_emoji_token,
    read_sentence_label_pair,
    repo_root,
    strip_wrapping_quotes,
)


def _rows(split: str):
    root = repo_root()
    raw = (root / "dataset" / f"{split}_sentence.csv").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    docs, labels = read_sentence_label_pair(
        root / "dataset" / f"{split}_sentence.csv",
        root / "dataset" / f"{split}_label.csv",
    )
    rows = []
    for line, doc, y in zip(raw, docs, labels):
        text = strip_wrapping_quotes(line)
        rows.append((text, int(y), any(is_emoji_token(t) for t in doc)))
    return rows


def main() -> int:
    test = _rows("test")
    sub = _rows("subtest")
    test_texts = {t for t, _, _ in test}
    sub_texts = {t for t, _, _ in sub}
    test_emoji = {t for t, _, e in test if e}

    in_test = sub_texts & test_texts
    only_sub = sub_texts - test_texts
    emoji_missing_from_sub = test_emoji - sub_texts

    print(f"test n={len(test)} unique={len(test_texts)}")
    print(f"subtest n={len(sub)} unique={len(sub_texts)}")
    print(f"subtest texts that also appear in test:     {len(in_test)}")
    print(f"subtest texts not in test:                  {len(only_sub)}")
    print(f"test emoji tweets missing from subtest:     {len(emoji_missing_from_sub)}")

    sub_emoji = sum(1 for _, _, e in sub if e)
    sub_pos_emoji = sum(1 for _, y, e in sub if e and y == 1)
    sub_neg_emoji = sum(1 for _, y, e in sub if e and y == 0)
    test_pos_emoji = sum(1 for _, y, e in test if e and y == 1)
    test_neg_emoji = sum(1 for _, y, e in test if e and y == 0)
    print()
    print(f"emoji tweets  test={test_pos_emoji + test_neg_emoji} "
          f"(pos {test_pos_emoji} / neg {test_neg_emoji})")
    print(f"emoji tweets  sub ={sub_emoji} "
          f"(pos {sub_pos_emoji} / neg {sub_neg_emoji})")
    if only_sub:
        print("\nsubtest-only examples (up to 5):")
        for text in list(only_sub)[:5]:
            print(" -", text[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
