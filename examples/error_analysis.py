#!/usr/bin/env python3
"""Show where the cue rules and bag-of-words NB disagree with the gold labels."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from sarcasm_lab.cue_model import RuleCueClassifier
from sarcasm_lab.cues import extract_cues
from sarcasm_lab.io import Split, load_all_splits
from sarcasm_lab.metrics import binary_metrics
from sarcasm_lab.naive_bayes import MultinomialNB
from sarcasm_lab.tables import format_table
from sarcasm_lab.vectorize import CountVectorizer


def _clip(text: str, n: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def _cases(split: Split, y_pred: list[int], want_gold: int, want_pred: int, limit: int = 4) -> list[str]:
    out = []
    for text, gold, pred in zip(split.texts, split.labels, y_pred):
        if gold == want_gold and pred == want_pred:
            out.append(_clip(text))
        if len(out) >= limit:
            break
    return out


def _confusion_row(name: str, y_true: list[int], y_pred: list[int]) -> list[str]:
    m = binary_metrics(y_true, y_pred)
    return [
        name,
        str(m.true_positive),
        str(m.false_positive),
        str(m.true_negative),
        str(m.false_negative),
    ]


def main() -> int:
    splits = load_all_splits(tokenize=True)
    train, test, subtest = splits["train"], splits["test"], splits["subtest"]

    vec = CountVectorizer(min_df=2, max_features=15000)
    X_train = vec.fit_transform(train.tokens)
    nb = MultinomialNB().fit(X_train, train.labels, vec.n_features)
    rule = RuleCueClassifier()

    test_cues = [extract_cues(t, tok) for t, tok in zip(test.texts, test.tokens)]
    sub_cues = [extract_cues(t, tok) for t, tok in zip(subtest.texts, subtest.tokens)]
    rule_test = rule.predict(test_cues)
    rule_sub = rule.predict(sub_cues)
    nb_test = nb.predict(vec.transform(test.tokens))

    print("Confusion counts (TP FP TN FN)")
    print(
        format_table(
            ["model", "TP", "FP", "TN", "FN"],
            [
                _confusion_row("rule / test", test.labels, rule_test),
                _confusion_row("rule / subtest", subtest.labels, rule_sub),
                _confusion_row("NB / test", test.labels, nb_test),
            ],
            right_align={1, 2, 3, 4},
        )
    )

    sarcastic_with_tag = sum(
        1 for gold, cues in zip(test.labels, test_cues) if gold == 1 and cues.has_sarcasm_hashtag
    )
    print()
    print(
        f"Sarcastic test tweets with a lexicon sarcasm hashtag: "
        f"{sarcastic_with_tag} / {sum(test.labels)} "
        f"({100 * sarcastic_with_tag / sum(test.labels):.1f}%)"
    )
    print("That is label leakage, not a model trick. #not on this collection is almost the label.")

    print()
    print("Rule false negatives on test (sarcastic, no lexicon fire)")
    for line in _cases(test, rule_test, want_gold=1, want_pred=0):
        print(f"  · {line}")
    print("Rule false positives on test (predicted sarcastic, labeled 0)")
    fps = _cases(test, rule_test, want_gold=0, want_pred=1)
    if fps:
        for line in fps:
            print(f"  · {line}")
    else:
        print("  · (none in the first scan — precision is near 100%)")

    print()
    print("NB false positives on test")
    for line in _cases(test, nb_test, want_gold=0, want_pred=1):
        print(f"  · {line}")
    print("NB false negatives on test")
    for line in _cases(test, nb_test, want_gold=1, want_pred=0):
        print(f"  · {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
