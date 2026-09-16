#!/usr/bin/env python3
"""Step through `ReadOpen` + Keras-style padding on a few tweets.

Does not load GloVe. Builds a toy word index from the selected lines
the same way `Preprocess` does (fit on texts, sequences, post-pad),
and prints the `count` vs `vocab+1` mismatch described in
docs/code-map.md.

    python examples/preprocess_walkthrough.py
    python examples/preprocess_walkthrough.py --split test --n 8
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset import load_split  # noqa: E402
from examples.lib.tokenize import comma_unwrap, pad_sequences, tokenize_tweet  # noqa: E402


def build_word_index(docs: Sequence[Sequence[str]]) -> Dict[str, int]:
    """Keras Tokenizer equivalent: rank by frequency, index from 1."""
    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(doc)
    # Keras 2 default: most common first, ties keep first-seen order.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {word: idx + 1 for idx, (word, _count) in enumerate(ranked)}


def texts_to_sequences(
    docs: Sequence[Sequence[str]], word_index: Dict[str, int]
) -> List[List[int]]:
    return [[word_index[tok] for tok in doc if tok in word_index] for doc in docs]


def choose_examples(split_name: str, n: int) -> List[tuple[str, int]]:
    split = load_split(split_name)
    # Prefer a mix: tagged sarcastic, untagged sarcastic, sincere + emoji.
    tagged = []
    untagged_pos = []
    neg = []
    for sentence, label in split.pairs():
        tokens = tokenize_tweet(sentence)
        tags = {tok for tok in tokens if tok.startswith("#")}
        sarc = tags & {"#not", "#sarcasm", "#sarcastic", "#sarcastictweet"}
        if label == 1 and sarc:
            tagged.append((sentence, label))
        elif label == 1:
            untagged_pos.append((sentence, label))
        else:
            neg.append((sentence, label))
    picked = tagged[: max(1, n // 3)] + untagged_pos[: max(1, n // 3)] + neg[: n]
    return picked[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "test", "subtest"))
    parser.add_argument("--n", type=int, default=6, help="how many tweets to show")
    args = parser.parse_args()

    pairs = choose_examples(args.split, args.n)
    tokenized = [tokenize_tweet(sentence) for sentence, _label in pairs]
    word_index = build_word_index(tokenized)
    sequences = texts_to_sequences(tokenized, word_index)
    maxlen = max((len(seq) for seq in sequences), default=0)
    padded = pad_sequences(sequences, maxlen)

    tweet_count = len(pairs)
    vocab_plus_one = len(word_index) + 1
    print(f"split              {args.split}")
    print(f"example tweets     {tweet_count}")
    print(f"word_index size    {len(word_index)}")
    print(f"maxlen (this toy)  {maxlen}")
    print(f"Preprocess count   {tweet_count}   (embedding rows used in 2023 code)")
    print(f"correct matrix     {vocab_plus_one}   (len(word_index)+1)")
    print(
        "The original Preprocess() sizes the embedding matrix to tweet "
        "count, not vocab+1. On the full train set tweet count (39780) "
        "is larger than the vocab, so the extra rows are just unused zeros."
    )
    print()

    inv = {idx: word for word, idx in word_index.items()}
    for i, ((sentence, label), tokens, seq, pad) in enumerate(
        zip(pairs, tokenized, sequences, padded), start=1
    ):
        print(f"--- example {i}  label={label} ---")
        print(f"raw line     {sentence}")
        print(f"comma unwrap {comma_unwrap(sentence)}")
        print(f"tokens       {tokens}")
        print(f"ids          {seq}")
        print(f"padded[{maxlen}] {pad}")
        decoded = [inv.get(idx, "<pad>") if idx else "<pad>" for idx in pad]
        print(f"decoded      {decoded}")
        print()

    print(
        "Next: examples/emoji_signal.py measures whether those emoji "
        "tokens actually correlate with the sarcastic label."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
