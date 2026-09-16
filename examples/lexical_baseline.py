#!/usr/bin/env python3
"""Train a from-scratch multinomial Naive Bayes model on tweet tokens.

This is a documentation baseline, not the course-project Keras model.
Hashtag supervision (`#not`, `#sarcasm`, …) is extremely predictive of the
stored labels. Pass ``--strip-supervision-tags`` to hide those tags.

Usage:
    python3 examples/lexical_baseline.py
    python3 examples/lexical_baseline.py --strip-supervision-tags
    python3 examples/lexical_baseline.py --show-errors 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from sarcasm_lib.dataset import load_all_splits
from sarcasm_lib.features import SUPERVISION_TAGS, has_supervision_tag
from sarcasm_lib.metrics import format_metrics
from sarcasm_lib.naive_bayes import MultinomialNB


def _print_top(model: MultinomialNB) -> None:
    print("tokens favoring class 1 (sarcastic):")
    for token, score in model.top_tokens(1, k=12):
        print(f"  {score:+.3f}  {token}")
    print("tokens favoring class 0:")
    for token, score in model.top_tokens(0, k=12):
        print(f"  {score:+.3f}  {token}")
    print()


def _print_errors(
    texts: list[str],
    labels: list[int],
    preds: list[int],
    *,
    limit: int,
) -> None:
    shown = 0
    print("disagreements:")
    for text, truth, pred in zip(texts, labels, preds):
        if truth == pred:
            continue
        tag = "has-supervision-tag" if has_supervision_tag(text) else "no-supervision-tag"
        snippet = text.replace("\n", " ")[:140]
        print(f"  gold={truth} pred={pred} [{tag}] {snippet}")
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        print("  (none)")
    print()


def _tag_rule_predict(texts: list[str]) -> list[int]:
    """Predict sarcastic iff a known supervision hashtag is present."""
    return [1 if has_supervision_tag(text) else 0 for text in texts]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strip-supervision-tags", action="store_true")
    parser.add_argument("--no-cues", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--show-errors", type=int, default=6)
    parser.add_argument("--faithful", action="store_true")
    args = parser.parse_args(argv)

    splits = load_all_splits(faithful=args.faithful)
    train = splits["train"]
    model = MultinomialNB(
        alpha=args.alpha,
        strip_supervision_tags=args.strip_supervision_tags,
        include_cues=not args.no_cues,
        min_count=args.min_count,
    )
    model.fit(train.sentences, train.labels)

    print("# Lexical Naive Bayes baseline")
    print(f"strip_supervision_tags = {args.strip_supervision_tags}")
    print(f"include_cues           = {not args.no_cues}")
    print(f"vocab size             = {len(model.vocab_)}")
    print(f"supervision tag set    = {sorted(SUPERVISION_TAGS)}")
    print()
    _print_top(model)

    for name in ("train", "test", "subtest"):
        split = splits[name]
        pred = model.predict(split.sentences)
        print(f"{name:8} NB     {format_metrics(split.labels, pred)}")
        rule = _tag_rule_predict(list(split.sentences))
        print(f"{name:8} #tag   {format_metrics(split.labels, rule)}")
    print()

    test = splits["test"]
    test_pred = model.predict(test.sentences)
    if args.show_errors:
        _print_errors(
            list(test.sentences),
            list(test.labels),
            test_pred,
            limit=args.show_errors,
        )

    tagged = sum(1 for text in test.sentences if has_supervision_tag(text))
    print(
        f"test tweets carrying a supervision hashtag: {tagged} / {len(test)} "
        f"({100.0 * tagged / len(test):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
