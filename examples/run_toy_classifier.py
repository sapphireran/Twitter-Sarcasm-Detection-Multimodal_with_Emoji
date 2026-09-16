#!/usr/bin/env python3
"""Fit a tiny logistic regressor on hash-embedded toy tweets.

Compares the 32-d text view with the 64-d concatenated view. This is *not* a
reproduction of the 2023 sklearn numbers — it is a readable proof that the
emoji half is usable as a feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.average_vectors import multimodal_features
from examples.hash_embeddings import default_tables
from examples.tokenize import tweet_tokenize
from examples.toy_classifier import fit_logreg, metrics, predict, predict_proba
from examples.toy_corpus import TOY_LABELS, TOY_TWEETS, illustrative_pair


def _line(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name:14s}  acc={stats['accuracy']:.3f}  "
        f"p={stats['precision']:.3f}  r={stats['recall']:.3f}  "
        f"f1={stats['f1']:.3f}"
    )


def main() -> int:
    text_table, emoji_table = default_tables(dim=32)
    tokenized = [tweet_tokenize(t) for t in TOY_TWEETS]
    x_text, x_multi = multimodal_features(tokenized, text_table, emoji_table)
    y = np.asarray(TOY_LABELS, dtype=int)

    text_model = fit_logreg(x_text, y, seed=1)
    multi_model = fit_logreg(x_multi, y, seed=1)
    pred_text = predict(text_model, x_text)
    pred_multi = predict(multi_model, x_multi)

    print("=== toy logistic regression (16 tweets, hash embeddings) ===")
    print(_line("text 32-d", metrics(y, pred_text)))
    print(_line("multi 64-d", metrics(y, pred_multi)))
    print(f"final loss text={text_model.losses[-1]:.4f}  multi={multi_model.losses[-1]:.4f}")

    sarcastic, literal = illustrative_pair()
    i_s = TOY_TWEETS.index(sarcastic)
    i_l = TOY_TWEETS.index(literal)
    p_text = predict_proba(text_model, x_text)
    p_multi = predict_proba(multi_model, x_multi)
    print()
    print("=== P(sarcastic) on the illustrative pair ===")
    print(f"{sarcastic!r}")
    print(f"  text {p_text[i_s]:.3f}   multi {p_multi[i_s]:.3f}")
    print(f"{literal!r}")
    print(f"  text {p_text[i_l]:.3f}   multi {p_multi[i_l]:.3f}")
    print(
        "The multimodal model should put more daylight between those two "
        "rows because only the emoji half differs."
    )

    print()
    print("=== per-tweet predictions (multi) ===")
    for text, gold, pred, prob in zip(TOY_TWEETS, y, pred_multi, p_multi):
        flag = "ok" if pred == gold else "MISS"
        print(f"  [{flag}] gold={gold} pred={pred} p={prob:.2f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
