#!/usr/bin/env python3
"""Tokenize a few real train / subtest rows and split text vs emoji channels.

This is the only example that prints original dataset lines. It is read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dataset_stats import SPLIT_FILES
from examples.hash_embeddings import default_tables
from examples.tokenize import read_pairs, tweet_tokenize


def _preview(split: str, n: int = 4) -> None:
    sent, lab = SPLIT_FILES[split]
    text_table, emoji_table = default_tables()
    print(f"=== {split} (first {n} rows) ===")
    for raw, label in read_pairs(sent, lab)[:n]:
        tokens = tweet_tokenize(raw)
        words = [tok for tok in tokens if text_table.known(tok)]
        faces = [tok for tok in tokens if emoji_table.known(tok)]
        tag = "SARC" if label == 1 else "lit "
        shown = raw if len(raw) <= 110 else raw[:107] + "..."
        print(f"[{tag}] {shown}")
        print(f"       tokens={len(tokens)}  text={words[:8]}{'…' if len(words) > 8 else ''}")
        print(f"       emoji={faces or '∅  (AverageVectorPerEmoji → zeros(200))'}")
    print()


def main() -> int:
    print("Real CSV tokenization — no embeddings loaded, no writes.\n")
    _preview("train", n=4)
    _preview("subtest", n=5)
    print(
        "Train rows often have an empty emoji channel. Subtest rows almost "
        "always have at least one pictograph, which is why docs/results.md "
        "shows the multimodal lift there."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
