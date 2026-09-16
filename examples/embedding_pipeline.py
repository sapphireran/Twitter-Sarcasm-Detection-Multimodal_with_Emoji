#!/usr/bin/env python3
"""Walk the GloVe / emoji2vec averaging recipe on a handful of real tweets.

The course notebooks load 200-d Twitter GloVe plus emoji2vec, then:

* ``AverageVectorPerTweet`` — mean of in-vocab word vectors (200-d)
* ``AverageVectorPerEmoji`` — mean of in-vocab emoji vectors (200-d)
* concatenate → 400-d features for SVM / trees / forests
* ``Preprocess`` — Keras tokenizer + post-padding + an embedding matrix
  that falls back to emoji2vec when a "word" is actually an emoji

This script repeats the control flow with a 16-d dummy table built from
the tokens on the current slice. Shapes and empty-tweet behaviour match
``data_utils.py`` (a tweet with no in-vocab tokens becomes a zero vector).

Usage:

    python3 examples/embedding_pipeline.py
    python3 examples/embedding_pipeline.py --split subtest --n 8 --dim 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset import load_split
from examples.lib.embeddings import (
    DummyKeyedVectors,
    average_vectors,
    build_padded_sequences,
    concatenate_modalities,
    emoji_token_predicate,
    tokenize_docs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="subtest", choices=("train", "test", "subtest"))
    parser.add_argument("--n", type=int, default=6, help="How many tweets to embed.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    split = load_split(args.split, replay_readopen=True)
    start = args.offset
    end = min(len(split), start + args.n)
    texts = list(split.texts[start:end])
    labels = list(split.labels[start:end])
    docs = tokenize_docs(texts)
    vocab_tokens = [token for doc in docs for token in doc]
    word_model = DummyKeyedVectors.random_from_tokens(vocab_tokens, dim=args.dim, seed=args.seed)
    emoji_model = DummyKeyedVectors.random_from_tokens(
        [token for token in vocab_tokens if emoji_token_predicate(token)],
        dim=args.dim,
        seed=args.seed + 1,
    )

    word_avg = average_vectors(docs, word_model)
    emoji_avg = average_vectors(docs, emoji_model, predicate=emoji_token_predicate)
    fused = concatenate_modalities(word_avg, emoji_avg)
    padded, word_index = build_padded_sequences(docs)

    print("Embedding pipeline walkthrough (dummy vectors, real tweets)")
    print(f"split={args.split} rows=[{start}:{end}] dim={args.dim}")
    print(f"dummy word vocab     {len(word_model.table)} tokens")
    print(f"dummy emoji vocab    {len(emoji_model.table)} tokens")
    print(f"word average shape   {word_avg.shape}   (GloVe path in ml_read_data)")
    print(f"emoji average shape  {emoji_avg.shape}   (AverageVectorPerEmoji)")
    print(f"concatenated shape   {fused.shape}   (sklearn multi-modal features)")
    print(f"padded sequences     {padded.shape}   word_index size={len(word_index)}")
    print()

    for i, (text, label, doc) in enumerate(zip(texts, labels, docs)):
        n_emoji = sum(1 for token in doc if emoji_token_predicate(token))
        word_norm = float(np.linalg.norm(word_avg[i]))
        emoji_norm = float(np.linalg.norm(emoji_avg[i]))
        preview = text.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        print(f"[{i}] label={label} tokens={len(doc)} emoji_tokens={n_emoji}")
        print(f"    {preview}")
        print(
            f"    |word|={word_norm:.3f}  |emoji|={emoji_norm:.3f}  "
            f"{'emoji zero-vector (no in-vocab emoji)' if emoji_norm == 0 else 'emoji signal present'}"
        )
        print(f"    padded ids: {padded[i].tolist()}")
        print()

    empty_docs = [[]]
    empty_avg = average_vectors(empty_docs, word_model)
    print(
        "Empty-tweet behaviour (matches data_utils.AverageVectorPerTweet): "
        f"shape={empty_avg.shape} all_zero={bool(np.all(empty_avg == 0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
