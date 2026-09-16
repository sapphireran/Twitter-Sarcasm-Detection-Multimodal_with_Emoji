#!/usr/bin/env python3
"""Tokens with the strongest class odds under the training-set Naive Bayes model."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sarcasm_lab.io import load_split
from sarcasm_lab.naive_bayes import MultinomialNB
from sarcasm_lab.tables import format_table
from sarcasm_lab.vectorize import CountVectorizer


def _top_odds(nb: MultinomialNB, vocab: dict[str, int], toward_sarcastic: bool, n: int = 15) -> list[tuple[str, float]]:
    if nb.classes_ != [0, 1]:
        raise RuntimeError(f"expected classes [0, 1], got {nb.classes_}")
    log0, log1 = nb.feature_log_prob_
    signed = []
    inv = {idx: tok for tok, idx in vocab.items()}
    for idx, token in inv.items():
        delta = log1[idx] - log0[idx]
        signed.append((token, delta))
    signed.sort(key=lambda item: item[1], reverse=toward_sarcastic)
    return signed[:n]


def main() -> int:
    train = load_split("train", tokenize=True)
    vec = CountVectorizer(min_df=4, max_features=12000)
    X = vec.fit_transform(train.tokens)
    nb = MultinomialNB(alpha=1.0).fit(X, train.labels, vec.n_features)

    sarcastic = _top_odds(nb, vec.vocabulary_, toward_sarcastic=True)
    sincere = _top_odds(nb, vec.vocabulary_, toward_sarcastic=False)

    def rows(pairs: list[tuple[str, float]]) -> list[list[str]]:
        return [[tok, f"{odds:.2f}"] for tok, odds in pairs]

    print("Highest log P(w | sarcastic) − log P(w | not)   (min_df=4)")
    print(format_table(["token", "log-odds"], rows(sarcastic), right_align={1}))
    print()
    print("Highest log P(w | not) − log P(w | sarcastic)")
    print(
        format_table(
            ["token", "log-odds"],
            [[tok, f"{-odds:.2f}"] for tok, odds in sincere],
            right_align={1},
        )
    )
    print()
    print("Hashtags used as distant-supervision labels (#not, #sarcastictweet,")
    print("#yeahright) dominate the sarcastic list. Low-count topical tags can")
    print("also float up; treat the long tail as hypothesis-generating, not causal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
