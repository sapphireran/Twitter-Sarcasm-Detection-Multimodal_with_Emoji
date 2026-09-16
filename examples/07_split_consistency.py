#!/usr/bin/env python3
"""Sanity-check the three official CSV pairs.

Failures here mean the docs (or a future edit) cannot trust line alignment.
The script exits non-zero if any check fails so it can be used as a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import read_sentence_label_pair, repo_root  # noqa: E402

EXPECTED = {
    "train": (39780, 18488, 21292),
    "test": (2000, 1000, 1000),
    "subtest": (278, 172, 106),
}


def main() -> int:
    root = repo_root()
    errors: list[str] = []
    for split, (n, pos, neg) in EXPECTED.items():
        sent = root / "dataset" / f"{split}_sentence.csv"
        lab = root / "dataset" / f"{split}_label.csv"
        if not sent.exists() or not lab.exists():
            errors.append(f"missing files for {split}")
            continue
        try:
            docs, labels = read_sentence_label_pair(sent, lab)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if len(docs) != n:
            errors.append(f"{split}: expected {n} tweets, got {len(docs)}")
        got_pos = int(labels.sum())
        got_neg = int((labels == 0).sum())
        if got_pos != pos or got_neg != neg:
            errors.append(
                f"{split}: expected pos/neg {pos}/{neg}, got {got_pos}/{got_neg}"
            )
        if not set(labels.tolist()).issubset({0, 1}):
            errors.append(f"{split}: labels outside {{0,1}}")
        empty = sum(1 for d in docs if len(d) == 0)
        if empty:
            errors.append(f"{split}: {empty} tweets tokenized to empty")
        print(
            f"ok  {split:<8} n={len(docs):5d} pos={got_pos:5d} "
            f"neg={got_neg:5d} empty_tok={empty}"
        )

    if errors:
        print("FAIL")
        for e in errors:
            print(" ", e)
        return 1
    print("all split consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
