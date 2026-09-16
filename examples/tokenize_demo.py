#!/usr/bin/env python3
"""Show the example tokenizer on a few committed tweets."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sarcasm_lab.cues import extract_cues
from sarcasm_lab.io import load_split
from sarcasm_lab.tokenize import tokenize_tweet


DEMOS = [
    "I loovee when people text back ... 😒 #sarcastictweet",
    "I love watching golf ! ⛳ ️ #not",
    "<user> Rest in peace & love to you and your family",
    "10k walk this morning. We did an awesome job.",
]


def _print_one(text: str, label: int | None = None) -> None:
    tokens = tokenize_tweet(text)
    cues = extract_cues(text, tokens)
    suffix = "" if label is None else f"   gold={label}"
    print(f"text:   {text}{suffix}")
    print(f"tokens: {tokens}")
    flags = []
    if cues.has_sarcasm_hashtag:
        flags.append("sarcasm-hashtag=" + ",".join(cues.sarcasm_hashtags))
    if cues.has_negative_emoji:
        flags.append("neg-emoji")
    if cues.has_positive_stem:
        flags.append("positive-stem")
    if cues.n_ellipsis:
        flags.append(f"ellipsis×{cues.n_ellipsis}")
    print("cues:   " + (", ".join(flags) if flags else "(none of the lexicon flags)"))
    print()


def main() -> int:
    print("Hand-picked strings (the first two appear on the subtest)\n")
    for text in DEMOS:
        _print_one(text)

    print("First three training rows as stored on disk\n")
    train = load_split("train", tokenize=True)
    for text, tokens, label in zip(train.texts[:3], train.tokens[:3], train.labels[:3]):
        _print_one(text, label=label)
        assert tokens == tokenize_tweet(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
