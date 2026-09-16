#!/usr/bin/env python3
"""Reprint the historical notebook metrics as tables.

Numbers are copied from get_metrics_of_models.ipynb and
baseline_models.ipynb outputs. This script does not load pickles or
SavedModels.

    python examples/report_metrics.py
    python examples/report_metrics.py --markdown
    python examples/report_metrics.py --csv
"""

from __future__ import annotations

import argparse
import csv
import io
from typing import Dict, List, Sequence

# Order inside each list: test W, test WE, sub W, sub WE
# Source: get_metrics_of_models.ipynb stored stdout / hardcoded dl_list.
METRICS: Dict[str, Dict[str, List[float]]] = {
    "decision_tree": {
        "accuracy": [0.7265, 0.7295, 0.7769784172661871, 0.7985611510791367],
        "f1": [0.7556945064761054, 0.7566351776878093, 0.8342245989304812, 0.8461538461538463],
        "recall": [0.846, 0.841, 0.9069767441860465, 0.8953488372093024],
        "precision": [0.6828087167070218, 0.687653311529027, 0.7722772277227723, 0.8020833333333334],
    },
    "svm": {
        "accuracy": [0.769, 0.763, 0.8129496402877698, 0.8237410071942446],
        "f1": [0.772189349112426, 0.7662721893491123, 0.8522727272727274, 0.8528528528528528],
        "recall": [0.783, 0.777, 0.872093023255814, 0.8255813953488372],
        "precision": [0.7616731517509727, 0.7558365758754864, 0.8333333333333334, 0.8819875776397516],
    },
    "random_forest": {
        "accuracy": [0.8145, 0.818, 0.8057553956834532, 0.8525179856115108],
        "f1": [0.8232491662696524, 0.8255033557046979, 0.8524590163934426, 0.8838526912181304],
        "recall": [0.864, 0.861, 0.9069767441860465, 0.9069767441860465],
        "precision": [0.7861692447679709, 0.7928176795580111, 0.8041237113402062, 0.861878453038674],
    },
    "gradient_boosting": {
        # Accuracy only; F1 was not stored in the metrics notebook.
        "accuracy": [0.746, 0.7475, 0.7949640287769785, 0.7949640287769785],
    },
    "bilstm_attention": {
        "accuracy": [0.8635, 0.8735, 0.8669064748201439, 0.8920863309352518],
        "f1": [0.8655834564254061, 0.8685714285714285, 0.8939828080229226, 0.9107142857142858],
        "recall": [0.879, 0.836, 0.9069767441860465, 0.8895348837209303],
        "precision": [0.8525703200775946, 0.9037837837837838, 0.8813559322033898, 0.9329268292682927],
    },
}

SETTINGS = ("test W", "test WE", "sub W", "sub WE")
MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "bilstm_attention": "Bi-LSTM + attention",
}


def we_minus_w(values: Sequence[float]) -> Dict[str, float]:
    return {"delta_test": values[1] - values[0], "delta_sub": values[3] - values[2]}


def render_plain() -> str:
    lines = [
        "Historical metrics from the 2023 notebooks (not a fresh eval).",
        "",
    ]
    for metric in ("accuracy", "f1", "recall", "precision"):
        lines.append(f"{metric}")
        header = f"{'model':<22}" + "".join(f"{col:>12}" for col in SETTINGS)
        lines.append(header)
        lines.append("-" * len(header))
        for key, label in MODEL_LABELS.items():
            row = METRICS[key].get(metric)
            if not row:
                lines.append(f"{label:<22}" + "".join(f"{'—':>12}" for _ in SETTINGS))
                continue
            lines.append(f"{label:<22}" + "".join(f"{v:12.4f}" for v in row))
        lines.append("")
    lines.append("WE − W accuracy")
    lines.append(f"{'model':<22}{'Δ test':>12}{'Δ sub':>12}")
    lines.append("-" * 46)
    for key, label in MODEL_LABELS.items():
        deltas = we_minus_w(METRICS[key]["accuracy"])
        lines.append(
            f"{label:<22}{deltas['delta_test']:+12.4f}{deltas['delta_sub']:+12.4f}"
        )
    lines.append("")
    lines.append("Headline number to cite: Bi-LSTM + attention, test WE, acc 0.8735 / F1 0.8686.")
    return "\n".join(lines)


def render_markdown() -> str:
    chunks = [
        "Historical metrics from the 2023 notebooks (not a fresh eval).",
        "",
    ]
    for metric in ("accuracy", "f1", "recall", "precision"):
        chunks.append(f"### {metric}")
        chunks.append("")
        chunks.append("| model | " + " | ".join(SETTINGS) + " |")
        chunks.append("| --- | " + " | ".join("---:" for _ in SETTINGS) + " |")
        for key, label in MODEL_LABELS.items():
            row = METRICS[key].get(metric)
            if not row:
                chunks.append("| " + label + " | " + " | ".join("—" for _ in SETTINGS) + " |")
            else:
                chunks.append(
                    "| "
                    + label
                    + " | "
                    + " | ".join(f"{v:.4f}" for v in row)
                    + " |"
                )
        chunks.append("")
    chunks.append("### WE − W accuracy")
    chunks.append("")
    chunks.append("| model | Δ test | Δ sub |")
    chunks.append("| --- | ---: | ---: |")
    for key, label in MODEL_LABELS.items():
        deltas = we_minus_w(METRICS[key]["accuracy"])
        chunks.append(
            f"| {label} | {deltas['delta_test']:+.4f} | {deltas['delta_sub']:+.4f} |"
        )
    chunks.append("")
    return "\n".join(chunks)


def render_csv() -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["model", "metric", *SETTINGS])
    for key, label in MODEL_LABELS.items():
        for metric, row in METRICS[key].items():
            writer.writerow([label, metric, *[f"{v:.10f}" for v in row]])
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--markdown", action="store_true")
    fmt.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    if args.markdown:
        print(render_markdown())
    elif args.csv:
        print(render_csv(), end="")
    else:
        print(render_plain())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
