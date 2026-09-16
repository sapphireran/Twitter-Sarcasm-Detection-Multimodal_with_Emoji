#!/usr/bin/env python3
"""Render docs/results/recorded_metrics.json as the markdown tables in the docs.

Keeping the JSON as the source of truth means a typo in experiments.md can
be caught by comparing this script's stdout to the checked-in page.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "docs" / "results" / "recorded_metrics.json"

MODELS = [
    ("svm", "SVM"),
    ("decision_tree", "Decision tree"),
    ("random_forest", "Random forest"),
    ("gradient_boosting", "Gradient boosting"),
    ("bilstm_attention", "BiLSTM + attention"),
]


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def table(metric: str) -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    slots = data["slot_order"]
    header = "| Model | " + " | ".join(slots) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(slots)) + " |"
    print(header)
    print(sep)
    for key, label in MODELS:
        block = data["models"][key]
        if metric not in block:
            continue
        cells = " | ".join(fmt(v) for v in block[metric])
        print(f"| {label} | {cells} |")
    print()


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    print(data["source"])
    print("slot order:", ", ".join(data["slot_order"]))
    print()
    print("## accuracy")
    table("accuracy")
    print("## f1")
    table("f1")
    print("## recall")
    table("recall")
    print("## precision")
    table("precision")
    keras = data["models"]["bilstm_attention"]["keras_evaluate"]
    print("## keras evaluate")
    for name, row in keras.items():
        print(f"  {name}: loss={row['loss']} acc={row['acc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
