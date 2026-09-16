"""Scores copied from the 2023 notebook outputs.

These are historical numbers, not re-runs. The Keras SavedModels in
``model/`` are missing their ``variables/`` shards, and several sklearn
pickles were never uploaded, so the archive cannot reproduce the
figures from the checked-in artifacts alone.

List order for classical models is
``[test W, test WE, subtest W, subtest WE]``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecordedRow:
    model: str
    modality: str  # "W" or "WE"
    split: str  # "test" or "subtest"
    accuracy: float
    f1: float
    precision: float | None = None
    recall: float | None = None
    source: str = "get_metrics_of_models.ipynb"


def _rows_from_lists(
    model: str,
    acc: list[float],
    f1: list[float],
    precision: list[float] | None = None,
    recall: list[float] | None = None,
) -> list[RecordedRow]:
    splits = ("test", "test", "subtest", "subtest")
    modalities = ("W", "WE", "W", "WE")
    rows: list[RecordedRow] = []
    for i, (split, modality) in enumerate(zip(splits, modalities)):
        rows.append(
            RecordedRow(
                model=model,
                modality=modality,
                split=split,
                accuracy=acc[i],
                f1=f1[i],
                precision=None if precision is None else precision[i],
                recall=None if recall is None else recall[i],
            )
        )
    return rows


RECORDED_ROWS: tuple[RecordedRow, ...] = tuple(
    _rows_from_lists(
        "SVM",
        [0.769, 0.763, 0.8129496402877698, 0.8237410071942446],
        [0.772189349112426, 0.7662721893491123, 0.8522727272727274, 0.8528528528528528],
        [0.7616731517509727, 0.7558365758754864, 0.8333333333333334, 0.8819875776397516],
        [0.783, 0.777, 0.872093023255814, 0.8255813953488372],
    )
    + _rows_from_lists(
        "DecisionTree",
        [0.7265, 0.7295, 0.7769784172661871, 0.7985611510791367],
        [0.7556945064761054, 0.7566351776878093, 0.8342245989304812, 0.8461538461538463],
        [0.6828087167070218, 0.687653311529027, 0.7722772277227723, 0.8020833333333334],
        [0.846, 0.841, 0.9069767441860465, 0.8953488372093024],
    )
    + _rows_from_lists(
        "RandomForest",
        [0.8145, 0.818, 0.8057553956834532, 0.8525179856115108],
        [0.8232491662696524, 0.8255033557046979, 0.8524590163934426, 0.8838526912181304],
        [0.7861692447679709, 0.7928176795580111, 0.8041237113402062, 0.861878453038674],
        [0.864, 0.861, 0.9069767441860465, 0.9069767441860465],
    )
    + _rows_from_lists(
        "GradientBoosting",
        [0.746, 0.7475, 0.7949640287769785, 0.7949640287769785],
        [0.75146771037182, 0.7528144884973079, 0.8376068376068376, 0.8357348703170029],
    )
    + [
        RecordedRow(
            "BiLSTM+Attn",
            "W",
            "test",
            0.8634999990463257,
            0.8655834564254061,
        ),
        RecordedRow(
            "BiLSTM+Attn",
            "WE",
            "test",
            0.8734999895095825,
            0.8685714285714285,
        ),
        RecordedRow(
            "BiLSTM+Attn",
            "W",
            "subtest",
            0.866906464099884,
            0.8939828080229226,
        ),
        RecordedRow(
            "BiLSTM+Attn",
            "WE",
            "subtest",
            0.8920863270759583,
            0.9107142857142858,
        ),
    ]
)


def rows_for(model: str | None = None, split: str | None = None) -> list[RecordedRow]:
    rows = list(RECORDED_ROWS)
    if model is not None:
        rows = [row for row in rows if row.model == model]
    if split is not None:
        rows = [row for row in rows if row.split == split]
    return rows


def markdown_table(rows: list[RecordedRow] | None = None) -> str:
    rows = list(RECORDED_ROWS) if rows is None else rows
    lines = [
        "| Model | Split | Modality | Accuracy | F1 | Precision | Recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        prec = "—" if row.precision is None else f"{row.precision:.4f}"
        rec = "—" if row.recall is None else f"{row.recall:.4f}"
        lines.append(
            f"| {row.model} | {row.split} | {row.modality} | "
            f"{row.accuracy:.4f} | {row.f1:.4f} | {prec} | {rec} |"
        )
    return "\n".join(lines)
