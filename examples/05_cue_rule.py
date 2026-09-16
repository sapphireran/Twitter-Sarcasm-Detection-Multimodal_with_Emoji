#!/usr/bin/env python3
"""Hashtag-only baseline versus the recorded 2023 SVM scores."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccs2lab.cues import cue_rule_label, profile_text
from ccs2lab.metrics import binary_scores
from ccs2lab.recorded import rows_for
from ccs2lab.report import markdown_table
from ccs2lab.slices import slice_scores
from ccs2lab.splits import load_bundle


def main() -> int:
    bundle = load_bundle()
    headers = ("split", "slice", "n", "accuracy", "precision", "recall", "f1")
    rows = []
    for split in bundle.all_splits():
        preds = [cue_rule_label(profile_text(text)) for text in split.texts]
        overall = binary_scores(split.labels, preds)
        rows.append(
            (
                split.name,
                "all",
                overall.n,
                overall.accuracy,
                overall.precision,
                overall.recall,
                overall.f1,
            )
        )
        for slice_row in slice_scores(list(split.texts), list(split.labels), preds):
            if slice_row.name == "all" or slice_row.scores is None:
                continue
            s = slice_row.scores
            rows.append(
                (
                    split.name,
                    slice_row.name,
                    slice_row.n,
                    s.accuracy,
                    s.precision,
                    s.recall,
                    s.f1,
                )
            )
    print("## cue rule (predict sarcastic iff an explicit hashtag is present)")
    print(markdown_table(headers, rows))
    print()
    print("## recorded SVM (notebook outputs, not a re-run)")
    svm_rows = []
    for rec in rows_for(model="SVM"):
        svm_rows.append(
            (
                rec.split,
                rec.modality,
                rec.accuracy,
                rec.f1,
                rec.precision if rec.precision is not None else float("nan"),
                rec.recall if rec.recall is not None else float("nan"),
            )
        )
    print(
        markdown_table(
            ("split", "modality", "accuracy", "f1", "precision", "recall"),
            svm_rows,
        )
    )
    print()
    print(
        "On test, the hashtag rule is a high-precision, moderate-recall "
        "classifier because `#not` / `#sarcasm` are dense. On the "
        "no_explicit_cue slice, recall is zero by construction. That gap "
        "is the part a sequence model still has to earn."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
