#!/usr/bin/env python3
"""Show how the 2023 comma-join changes tokens vs keeping punctuation.

``data_utils.ReadOpen`` does ``' '.join(line.split(','))`` before NLTK
tokenizes. That drops commas that were part of the tweet. The lite tokenizer
can do either. This script prints both views for the tiny fixture and for
one real test-line that contains commas.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lite_pipeline import (  # noqa: E402
    mimic_readopen_commas,
    read_sentence_label_pair,
    strip_wrapping_quotes,
    tokenize,
)

FIXTURE_S = Path(__file__).resolve().parent / "fixtures" / "tiny_sentence.csv"
FIXTURE_L = Path(__file__).resolve().parent / "fixtures" / "tiny_label.csv"


def show(title: str, raw: str) -> None:
    cleaned = strip_wrapping_quotes(raw)
    keep = tokenize(cleaned)
    smashed = tokenize(mimic_readopen_commas(cleaned))
    print(f"--- {title}")
    print(f"raw:    {raw}")
    print(f"keep:   {keep}")
    print(f"smash:  {smashed}")
    if keep != smashed:
        missing = [t for t in keep if t not in smashed]
        print(f"delta:  tokens only in keep-commas view: {missing}")
    print()


def main() -> int:
    docs_keep, labels = read_sentence_label_pair(FIXTURE_S, FIXTURE_L, join_commas=False)
    docs_smash, _ = read_sentence_label_pair(FIXTURE_S, FIXTURE_L, join_commas=True)
    print(f"fixture tweets: {len(labels)}  sarcastic: {int(labels.sum())}")
    print()
    raw_lines = FIXTURE_S.read_text(encoding="utf-8").splitlines()
    for raw, keep, smash, y in zip(raw_lines, docs_keep, docs_smash, labels):
        print(f"label={int(y)}  keep={keep}")
        if keep != smash:
            print(f"         smash={smash}")
    print()
    # A real test tweet that is quoted because of commas.
    test_path = ROOT / "dataset" / "test_sentence.csv"
    if test_path.exists():
        for line in test_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "," in line:
                show("first comma-bearing test line", line)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
