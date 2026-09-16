"""Recorded 2023 metrics plus tiny helpers for the rule baseline.

Numbers come from ``get_metrics_of_models.ipynb`` (executed 5–6 June 2023)
and ``evaluate_loaded_dl_models.ipynb``. Column order is always:

0. single-modal, official test (2,000 tweets)
1. multi-modal, official test
2. single-modal, emoji-rich subtest (278 tweets)
3. multi-modal, emoji-rich subtest

``W`` in the notebooks means word/GloVe only. ``WE`` means word + emoji2vec.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Rounded percentages as printed in the notebook's final export cell, plus
# the raw floats kept for anyone who wants more than two decimals.
RECORDED_RESULTS: dict[str, dict[str, object]] = {
    "source": "get_metrics_of_models.ipynb + evaluate_loaded_dl_models.ipynb (June 2023)",
    "columns": (
        "single_modal_test",
        "multi_modal_test",
        "single_modal_subtest",
        "multi_modal_subtest",
    ),
    "splits": {
        "train": {"n": 39780, "sarcastic": 18488, "non_sarcastic": 21292},
        "test": {"n": 2000, "sarcastic": 1000, "non_sarcastic": 1000},
        "subtest": {"n": 278, "sarcastic": 172, "non_sarcastic": 106},
    },
    "models": {
        "svm": {
            "label": "SVM",
            "accuracy": [0.769, 0.763, 0.8129496402877698, 0.8237410071942446],
            "f1": [0.772189349112426, 0.7662721893491123, 0.8522727272727274, 0.8528528528528528],
            "recall": [0.783, 0.777, 0.872093023255814, 0.8255813953488372],
            "precision": [0.7616731517509727, 0.7558365758754864, 0.8333333333333334, 0.8819875776397516],
        },
        "dt": {
            "label": "Decision Tree",
            "accuracy": [0.7265, 0.7295, 0.7769784172661871, 0.7985611510791367],
            "f1": [0.7556945064761054, 0.7566351776878093, 0.8342245989304812, 0.8461538461538463],
            "recall": [0.846, 0.841, 0.9069767441860465, 0.8953488372093024],
            "precision": [0.6828087167070218, 0.687653311529027, 0.7722772277227723, 0.8020833333333334],
        },
        "rf": {
            "label": "Random Forest",
            "accuracy": [0.8145, 0.818, 0.8057553956834532, 0.8525179856115108],
            "f1": [0.8232491662696524, 0.8255033557046979, 0.8524590163934426, 0.8838526912181304],
            "recall": [0.864, 0.861, 0.9069767441860465, 0.9069767441860465],
            "precision": [0.7861692447679709, 0.7928176795580111, 0.8041237113402062, 0.861878453038674],
        },
        "gbt": {
            "label": "Gradient Boosting",
            "accuracy": [0.746, 0.7475, 0.7949640287769785, 0.7949640287769785],
            "f1": [0.75146771037182, 0.7528144884973079, 0.8376068376068376, 0.8357348703170029],
            "recall": None,
            "precision": None,
        },
        "bilstm_att": {
            "label": "BiLSTM + Attention",
            "accuracy": [0.8635, 0.8735, 0.8669064748201439, 0.8920863309352518],
            "f1": [0.8655834564254061, 0.8685714285714285, 0.8939828080229226, 0.9107142857142858],
            "recall": [0.879, 0.836, 0.9069767441860465, 0.8895348837209303],
            "precision": [0.8525703200775946, 0.9037837837837838, 0.8813559322033898, 0.9329268292682927],
        },
    },
}


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}"


def format_results_table(
    metric: str = "accuracy",
    *,
    results: dict[str, dict[str, object]] | None = None,
) -> str:
    """Render a Markdown table for one metric across every recorded model."""

    payload = results or RECORDED_RESULTS
    headers = [
        "Model",
        "W test",
        "WE test",
        "W subtest",
        "WE subtest",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    models = payload["models"]
    assert isinstance(models, dict)
    for key in ("svm", "dt", "rf", "gbt", "bilstm_att"):
        model = models[key]
        values = model.get(metric)
        label = str(model["label"])
        if values is None:
            cells = [label] + ["—"] * 4
        else:
            cells = [label] + [pct(float(v)) for v in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        return 0.0
    hits = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return hits / len(y_true)


def _counts(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp, fp, _, _ = _counts(y_true, y_pred)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp, _, _, fn = _counts(y_true, y_pred)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def metric_bundle(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    truth = list(y_true)
    pred = list(y_pred)
    return {
        "accuracy": accuracy(truth, pred),
        "precision": precision_score(truth, pred),
        "recall": recall_score(truth, pred),
        "f1": f1_score(truth, pred),
    }
