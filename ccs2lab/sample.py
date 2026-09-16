"""Sampling helpers.

The checked-in CSVs are **block-sorted by label**:

* train: long sincere runs, then almost all sarcastic rows at the end
  (only 13 label switches in 39,780 lines)
* test: 1,000 sarcastic then 1,000 sincere
* subtest: 172 sarcastic then 106 sincere

A prefix ``--train-limit`` is therefore not a subsample. It can be
99.9% sincere, and ``#not`` in that prefix is the noisy
``#not ready yet`` case. Always shuffle or stratify before cutting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def stratified_take(
    texts: Sequence[str],
    labels: Sequence[int],
    limit: int,
    *,
    seed: int = 7,
) -> tuple[list[str], list[int]]:
    """Return up to ``limit`` rows, preserving the source class rate."""
    if limit <= 0 or limit >= len(texts):
        return list(texts), list(labels)
    if len(texts) != len(labels):
        raise ValueError("texts and labels must align")
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[int]] = {0: [], 1: []}
    for index, label in enumerate(labels):
        buckets[int(label)].append(index)
    n_pos = int(round(limit * (len(buckets[1]) / len(labels))))
    n_pos = min(max(n_pos, 0), len(buckets[1]), limit)
    n_neg = min(limit - n_pos, len(buckets[0]))
    chosen = []
    if n_neg:
        chosen.extend(rng.choice(buckets[0], size=n_neg, replace=False).tolist())
    if n_pos:
        chosen.extend(rng.choice(buckets[1], size=n_pos, replace=False).tolist())
    rng.shuffle(chosen)
    return [texts[i] for i in chosen], [labels[i] for i in chosen]
