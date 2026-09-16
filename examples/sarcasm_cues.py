#!/usr/bin/env python3
"""Score a transparent hashtag / contrast rule against each official split.

The BiLSTM does not use these rules. They exist so the documented 87%
test accuracy has a lower bound: ``#not`` is almost definitional in this
dump, but most sarcastic test tweets do not carry it.

Usage:

    python3 examples/sarcasm_cues.py
    python3 examples/sarcasm_cues.py --split test --show-errors 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.cues import SARCASM_HASHTAGS, extract_cues, rule_predict
from examples.lib.dataset import SPLITS, load_split
from examples.lib.metrics import metric_bundle
from examples.lib.tokenize import extract_hashtags


def evaluate_split(name: str, show_errors: int) -> dict[str, object]:
    split = load_split(name)
    predictions = []
    reasons = Counter()
    tag_hits = Counter()
    tag_totals = Counter()
    false_pos: list[tuple[str, str]] = []
    false_neg: list[tuple[str, str]] = []

    for text, label in zip(split.texts, split.labels):
        decision = rule_predict(text)
        predictions.append(decision.label)
        reasons[decision.reason] += 1
        for tag in extract_hashtags(text):
            tag_totals[tag] += 1
            if label == 1:
                tag_hits[tag] += 1
        if decision.label == 1 and label == 0 and len(false_pos) < show_errors:
            false_pos.append((decision.reason, text))
        if decision.label == 0 and label == 1 and len(false_neg) < show_errors:
            false_neg.append((decision.reason, text))

    metrics = metric_bundle(split.labels, predictions)
    marker_coverage = sum(
        1
        for text, label in zip(split.texts, split.labels)
        if label == 1 and extract_cues(text).has_explicit_marker
    )
    n_sarc = sum(split.labels)
    print(f"## {name}  (n={len(split)})")
    print(
        "  rule metrics     "
        + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items())
    )
    print(
        f"  sarcastic tweets with an explicit marker  "
        f"{marker_coverage}/{n_sarc} ({(marker_coverage / n_sarc) if n_sarc else 0:.3f})"
    )
    print("  decision reasons " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))

    interesting = [
        tag
        for tag, _ in tag_totals.most_common(40)
        if tag in SARCASM_HASHTAGS or tag_totals[tag] >= 8
    ]
    if interesting:
        print("  P(sarcastic | hashtag) for frequent / marker tags:")
        for tag in interesting[:12]:
            tot = tag_totals[tag]
            print(f"    #{tag:<18} n={tot:4d}  p={tag_hits[tag] / tot:.3f}")

    def _preview(title: str, rows: list[tuple[str, str]]) -> None:
        if not rows:
            return
        print(f"  {title}")
        for reason, text in rows:
            preview = text.replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:97] + "..."
            print(f"    [{reason}] {preview}")

    _preview("false positives", false_pos)
    _preview("false negatives (sarcastic, no firing rule)", false_neg)
    print()
    return {
        "split": name,
        "metrics": metrics,
        "reasons": dict(reasons),
        "explicit_marker_coverage": marker_coverage / n_sarc if n_sarc else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", choices=list(SPLITS))
    parser.add_argument("--show-errors", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    names = tuple(args.split) if args.split else SPLITS
    print("Surface-cue rule baseline")
    print(
        "Predict sarcastic if the tweet has an explicit sarcasm hashtag "
        f"({sorted(SARCASM_HASHTAGS)}) or a positive word plus a groan emoji.\n"
    )
    reports = [evaluate_split(name, args.show_errors) for name in names]
    if args.json:
        print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
