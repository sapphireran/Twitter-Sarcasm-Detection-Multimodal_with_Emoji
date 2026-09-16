"""Notebook-reported numbers for the 2023 UCPH CCS2 project.

These values are copied from the executed cells in
``get_metrics_of_models.ipynb`` / ``baseline_models.ipynb``. Examples
compare the cue baselines against this table so readers can see the
gap the BiLSTM actually closed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .paths import REPORTED_RESULTS_PATH

# Fallback copy so imports still work if the JSON file is missing.
REPORTED: dict[str, Any] = {
    "source": "get_metrics_of_models.ipynb (executed 2023-06-05/06)",
    "splits": {
        "test": {"n": 2000, "sarcastic": 1000, "literal": 1000},
        "subtest": {"n": 278, "sarcastic": 172, "literal": 106},
    },
    "models": {
        "svm": {
            "test": {"single": {"acc": 0.7690, "f1": 0.7722}, "multi": {"acc": 0.7630, "f1": 0.7663}},
            "subtest": {"single": {"acc": 0.8129, "f1": 0.8523}, "multi": {"acc": 0.8237, "f1": 0.8529}},
        },
        "decision_tree": {
            "test": {"single": {"acc": 0.7265, "f1": 0.7557}, "multi": {"acc": 0.7295, "f1": 0.7566}},
            "subtest": {"single": {"acc": 0.7770, "f1": 0.8342}, "multi": {"acc": 0.7986, "f1": 0.8462}},
        },
        "random_forest": {
            "test": {"single": {"acc": 0.8145, "f1": 0.8232}, "multi": {"acc": 0.8180, "f1": 0.8255}},
            "subtest": {"single": {"acc": 0.8058, "f1": 0.8525}, "multi": {"acc": 0.8525, "f1": 0.8839}},
        },
        "gradient_boosting": {
            "test": {"single": {"acc": 0.7460, "f1": 0.7515}, "multi": {"acc": 0.7475, "f1": 0.7528}},
            "subtest": {"single": {"acc": 0.7950, "f1": 0.8376}, "multi": {"acc": 0.7950, "f1": 0.8357}},
        },
        "bilstm_attention": {
            "test": {"single": {"acc": 0.8635, "f1": 0.8656}, "multi": {"acc": 0.8735, "f1": 0.8686}},
            "subtest": {"single": {"acc": 0.8669, "f1": 0.8940}, "multi": {"acc": 0.8921, "f1": 0.9107}},
        },
    },
}


@lru_cache(maxsize=1)
def load_reported() -> dict[str, Any]:
    if REPORTED_RESULTS_PATH.is_file():
        return json.loads(REPORTED_RESULTS_PATH.read_text(encoding="utf-8"))
    return REPORTED


def iter_reported_rows() -> list[dict[str, Any]]:
    table = load_reported()
    rows: list[dict[str, Any]] = []
    for model, splits in table["models"].items():
        for split, modes in splits.items():
            for mode, scores in modes.items():
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "mode": mode,
                        "acc": scores["acc"],
                        "f1": scores["f1"],
                    }
                )
    return rows
