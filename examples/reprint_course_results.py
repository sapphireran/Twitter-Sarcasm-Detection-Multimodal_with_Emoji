#!/usr/bin/env python3
"""Print the June 2023 notebook metrics as a text table.

Source: executed cells in get_metrics_of_models.ipynb and
evaluate_loaded_dl_models.ipynb. This script does not load pickles or
SavedModels. It exists so the evaluation write-up stays honest about
which numbers are historical.
"""

from __future__ import annotations

import argparse
from textwrap import dedent

# Column order matches the notebooks:
#   0 single-modal test
#   1 multi-modal test
#   2 single-modal subtest
#   3 multi-modal subtest
#
# Each model stores [acc, f1, recall, precision] lists. None = not recorded.

RESULTS = {
    "SVM": {
        "acc": [0.769, 0.763, 0.8129496402877698, 0.8237410071942446],
        "f1": [0.772189349112426, 0.7662721893491123, 0.8522727272727274, 0.8528528528528528],
        "rec": [0.783, 0.777, 0.872093023255814, 0.8255813953488372],
        "prec": [0.7616731517509727, 0.7558365758754864, 0.8333333333333334, 0.8819875776397516],
    },
    "Decision tree": {
        "acc": [0.7265, 0.7295, 0.7769784172661871, 0.7985611510791367],
        "f1": [0.7556945064761054, 0.7566351776878093, 0.8342245989304812, 0.8461538461538463],
        "rec": [0.846, 0.841, 0.9069767441860465, 0.8953488372093024],
        "prec": [0.6828087167070218, 0.687653311529027, 0.7722772277227723, 0.8020833333333334],
        "notes": "multi-modal pickle may be an SVC; see docs/training.md",
    },
    "Random forest": {
        "acc": [0.8145, 0.818, 0.8057553956834532, 0.8525179856115108],
        "f1": [0.8232491662696524, 0.8255033557046979, 0.8524590163934426, 0.8838526912181304],
        "rec": [0.864, 0.861, 0.9069767441860465, 0.9069767441860465],
        "prec": [0.7861692447679709, 0.7928176795580111, 0.8041237113402062, 0.861878453038674],
    },
    "Gradient boosting": {
        "acc": [0.746, 0.7475, 0.7949640287769785, 0.7949640287769785],
        "f1": [0.75146771037182, 0.7528144884973079, 0.8376068376068376, 0.8357348703170029],
        "rec": [None, None, None, None],
        "prec": [None, None, None, None],
    },
    "Bi-LSTM + Attention": {
        "acc": [0.8635, 0.8735, 0.8669064748201439, 0.8920863309352518],
        "f1": [0.8655834564254061, 0.8685714285714285, 0.8939828080229226, 0.9107142857142858],
        "rec": [0.879, 0.836, 0.9069767441860465, 0.8895348837209303],
        "prec": [0.8525703200775946, 0.9037837837837838, 0.8813559322033898, 0.9329268292682927],
    },
}

COLUMNS = (
    (0, "test / text"),
    (1, "test / +emoji"),
    (2, "subtest / text"),
    (3, "subtest / +emoji"),
)


def fmt(value) -> str:
    if value is None:
        return "   —  "
    return f"{value:6.3f}"


def table(metric: str) -> str:
    header = f"{'model':22s}" + "".join(f"  {title:>16s}" for _, title in COLUMNS)
    lines = [header, "-" * len(header)]
    for name, blob in RESULTS.items():
        row = f"{name:22s}"
        values = blob[metric]
        for idx, _ in COLUMNS:
            row += f"  {fmt(values[idx]):>16s}"
        lines.append(row)
    return "\n".join(lines)


def deltas() -> str:
    """Multi-modal minus single-modal accuracy, per split."""
    lines = ["accuracy gain from adding emoji2vec (multi − single)", "-" * 56]
    for name, blob in RESULTS.items():
        acc = blob["acc"]
        test_d = acc[1] - acc[0]
        sub_d = acc[3] - acc[2]
        lines.append(f"  {name:22s}  test {test_d:+.3f}   subtest {sub_d:+.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        choices=("acc", "f1", "rec", "prec", "all"),
        default="all",
    )
    args = parser.parse_args(argv)

    print(
        dedent(
            """\
            Historical metrics from the 2023 CCS2 notebooks.
            Positive class = sarcastic (label 1).
            Source cells are listed in docs/evaluation.md.
            """
        )
    )
    metrics = ("acc", "f1", "rec", "prec") if args.metric == "all" else (args.metric,)
    titles = {
        "acc": "Accuracy",
        "f1": "F1 (sarcastic)",
        "rec": "Recall (sarcastic)",
        "prec": "Precision (sarcastic)",
    }
    for metric in metrics:
        print(titles[metric])
        print(table(metric))
        print()
    print(deltas())
    print()
    print(
        "Notes: Decision-tree +emoji may be an SVC (notebook typo). "
        "GBT P/R were not stored in the executed cells. "
        "Bi-LSTM +emoji is a precision gain on test (0.853 → 0.904) "
        "and an accuracy gain on the emoji subtest (+0.025)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
