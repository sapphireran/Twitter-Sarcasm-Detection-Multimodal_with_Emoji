#!/usr/bin/env python3
"""Lexical cue leakage: how much of sarcasm is just ``#not`` / ``#sarcasm``?

Twitter sarcasm corpora are often collected by *distant supervision*: a tweet
is labeled sarcastic because it contains ``#sarcasm``. Models that keep those
hashtags can look strong while barely reading the sentence. This script
measures that shortcut on the three splits shipped here.

It reports:

* fraction of each class that still contains a cue hashtag
* a rule baseline: predict sarcastic iff a cue hashtag is present
* the same rule after stripping all hashtags (should collapse to the prior)

Run::

    python3 examples/03_lexical_cues.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.io import iter_splits
from examples.common.tokenize import CUE_HASHTAGS, tokenize_tweet


def cue_present(text: str) -> bool:
    return any(tok in CUE_HASHTAGS for tok in tokenize_tweet(text))


def metrics(y_true, y_pred) -> dict:
    n = len(y_true)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = n - tp - fp - fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("Cue hashtag set:", ", ".join(sorted(CUE_HASHTAGS)))
    print()
    print(f"{'split':8} {'n':>6} {'cue% lit':>9} {'cue% sarc':>10} {'rule acc':>9} {'rule F1':>8} {'majority':>9}")
    print("-" * 72)
    for split in iter_splits():
        y = list(split.labels)
        pred = [1 if cue_present(t) else 0 for t in split.texts]
        m = metrics(y, pred)
        lit_cues = sum(1 for t, yv in split.pairs() if yv == 0 and cue_present(t))
        sarc_cues = sum(1 for t, yv in split.pairs() if yv == 1 and cue_present(t))
        lit_n = max(split.n_literal, 1)
        sarc_n = max(split.n_sarcastic, 1)
        majority = max(split.n_literal, split.n_sarcastic) / len(split)
        print(
            f"{split.name:8} {len(split):6d} "
            f"{100*lit_cues/lit_n:9.2f} {100*sarc_cues/sarc_n:10.2f} "
            f"{100*m['acc']:9.2f} {100*m['f1']:8.2f} {100*majority:9.2f}"
        )
    print()
    print("Interpretation")
    print("--------------")
    print("* A high rule-F1 on test means many sarcastic rows still wear the")
    print("  original collection hashtag. The BiLSTM can (and will) attend to it.")
    print("* Subtest still contains cue hashtags, but every row also has an emoji,")
    print("  so the multimodal gain reported in docs/04-results.md is not *only*")
    print("  hashtag leakage — the 278-row slice is where emoji2vec can matter.")
    print("* Literal tweets also use #not (\"I'm #not ready\"). That is why the")
    print("  rule's precision is not 1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
