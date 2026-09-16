#!/usr/bin/env python3
"""End-to-end example pipeline on the local CSVs.

Steps:

1. Load train/test/subtest
2. Tokenize a sarcastic tweet and a literal tweet
3. Build cue features
4. Score with the lexicon baseline
5. Fit the cue logistic on a training subset
6. Print metrics next to the 2023 notebook table

This is the script to run if you want a single command that exercises
the docs/examples stack without TensorFlow or GloVe.
"""

from __future__ import annotations

import json
from pathlib import Path

import _path  # noqa: F401

from sarcasm_toolkit.attention import attention_report
from sarcasm_toolkit.baseline import CueLogistic, LexiconBaseline, describe_prediction
from sarcasm_toolkit.dataset import load_split, summarize_split
from sarcasm_toolkit.embeddings import embed_tokens
from sarcasm_toolkit.metrics import binary_metrics, format_metrics, majority_baseline
from sarcasm_toolkit.reported import format_reported_table
from sarcasm_toolkit.tokenize import tokenize_tweet

ATTENTION_W = [0.05, 0.05, 1.40, 1.10, 0.15, 0.05, 0.05, 0.05]
OUT_PATH = Path(__file__).resolve().parent / "pipeline_last_run.json"


def pick_examples(split_name: str) -> tuple[str, str]:
    split = load_split(split_name)
    sarcastic = next(text for text, label in zip(split.texts, split.labels) if label == 1)
    literal = next(text for text, label in zip(split.texts, split.labels) if label == 0)
    return sarcastic, literal


def main() -> None:
    print("Pipeline walkthrough")
    print("====================")
    summaries = {name: summarize_split(load_split(name)) for name in ("train", "test", "subtest")}
    for name, summary in summaries.items():
        print(
            f"{name:8s} n={summary['n']:5d} sarcastic_rate={summary['sarcastic_rate']:.3f}"
        )

    sarcastic, literal = pick_examples("subtest")
    print("\nSubtest pair")
    print(f"  sarcastic: {sarcastic}")
    print(f"  literal:   {literal}")

    for title, text in (("sarcastic", sarcastic), ("literal", literal)):
        tokens = tokenize_tweet(text)
        report = attention_report(embed_tokens(tokens), ATTENTION_W, tokens=tokens)
        top = max(report, key=lambda row: row["weight"])
        print(f"  {title} top attention token: {top['token']!r} ({top['weight']:.3f})")

    train = load_split("train")
    model = CueLogistic(epochs=25, learning_rate=0.3)
    # 3k rows keeps the walkthrough snappy while still fitting cue weights.
    model.fit(train.texts[:3000], train.labels[:3000])
    lexicon = LexiconBaseline()

    payload = {"summaries": summaries, "metrics": {}}
    print("\nMetrics")
    for name in ("test", "subtest"):
        split = load_split(name)
        log_m = binary_metrics(split.labels, model.predict(split.texts))
        lex_m = binary_metrics(split.labels, lexicon.predict(split.texts))
        maj_m = binary_metrics(split.labels, majority_baseline(split.labels))
        print(format_metrics(f"logistic/{name}", log_m))
        print(format_metrics(f"lexicon/{name} ", lex_m))
        print(format_metrics(f"majority/{name}", maj_m))
        payload["metrics"][name] = {
            "logistic": log_m.as_dict(),
            "lexicon": lex_m.as_dict(),
            "majority": maj_m.as_dict(),
        }

    print("\nDescribed predictions")
    for text in (sarcastic, literal):
        info = describe_prediction(text, model)
        print(
            f"  pred={info['prediction']} p={info['probability']:.3f} "
            f"cues={info['cue_hashtags']} :: {text[:90]}"
        )

    payload["weights"] = model.top_weights(8)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH.name}")
    print()
    print(format_reported_table())


if __name__ == "__main__":
    main()
