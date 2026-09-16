#!/usr/bin/env python3
"""Mean-pool GloVe + mean-pool emoji2vec, then concatenate.

This is the classical multi-modal path in `data_utils.ml_read_data`,
run on a hand-built 8-d table so it does not need the real GloVe dump.

Usage:

    python3 examples/toy_embedding_fusion.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.embeddings import fuse_modalities, toy_emoji, toy_glove  # noqa: E402
from examples.lib.tweet_tokenize import tokenize_tweet  # noqa: E402

EXAMPLES = (
    (1, "I loovee when people text back ... 😒 #sarcastictweet"),
    (1, "Oh how I love getting home from work at 3am and my house being dirty #not"),
    (1, "I love walking to school 😄 #SarcasticTweet"),
    (0, "Want to have someone to speak to I'm so bored 😭"),
    (0, "i just imagined you dancing like this"),
)


def _fmt(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.2f}" for v in vec) + "]"


def explain(label: int, sentence: str) -> None:
    tokens = tokenize_tweet(sentence)
    word_table, emoji_table = toy_glove(), toy_emoji()
    word_mean, emoji_mean, fused = fuse_modalities(tokens, word_table, emoji_table)
    in_word = [t for t in tokens if t in word_table]
    in_emoji = [t for t in tokens if t in emoji_table]
    missed = [t for t in tokens if t not in word_table and t not in emoji_table]
    print(f"[{label}] {sentence}")
    print(f"     tokens:     {tokens}")
    print(f"     in GloVe:   {in_word or '—'}")
    print(f"     in emoji:   {in_emoji or '—'}")
    if missed:
        print(f"     dropped:    {missed}")
    print(f"     word mean:  {_fmt(word_mean)}")
    print(f"     emoji mean: {_fmt(emoji_mean)}")
    print(f"     concat[{len(fused)}]: {_fmt(fused)}")
    # Axis 2 of the toy GloVe is the sarcasm-tag channel; axis 0 of
    # the toy emoji table is the negative-face channel.
    print(
        f"     tag axis   {word_mean[2]:+.2f}   "
        f"neg-emoji axis {emoji_mean[0]:+.2f}"
    )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print("Toy tables are 8-d. Real coursework tables are 200-d + 200-d.")
    print("Fusion is still mean(word rows) ∥ mean(emoji rows).\n")
    for label, sentence in EXAMPLES:
        explain(label, sentence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
