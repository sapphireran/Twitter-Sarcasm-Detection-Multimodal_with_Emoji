"""Frozen June 2023 metric cards, transcribed from the executed notebooks.

Source cells live in ``get_metrics_of_models.ipynb`` (the ``svm_list`` /
``dt_list`` / ``rf_list`` / ``dl_list`` assignments) plus the named
gradient-boosting prints. Docs and ``reported_results`` both import this
module so a typo has to be fixed once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

CONDITIONS: Tuple[str, ...] = (
    "test_word",
    "test_word_emoji",
    "subtest_word",
    "subtest_word_emoji",
)

CONDITION_LABELS = {
    "test_word": "Test · word",
    "test_word_emoji": "Test · word+emoji",
    "subtest_word": "Subtest · word",
    "subtest_word_emoji": "Subtest · word+emoji",
}


@dataclass(frozen=True)
class MetricCard:
    model: str
    accuracy: Tuple[float, float, float, float]
    f1: Tuple[float, float, float, float]
    recall: Tuple[float, float, float, float] | None = None
    precision: Tuple[float, float, float, float] | None = None

    def as_row(self, metric: str) -> Tuple[float, ...]:
        values = getattr(self, metric)
        if values is None:
            raise KeyError(f"{self.model} has no {metric} recorded")
        return values

    def percent(self, metric: str, digits: int = 2) -> List[str]:
        return [f"{value * 100:.{digits}f}" for value in self.as_row(metric)]


# Fractions copied from the notebook, not re-rounded.
SVM = MetricCard(
    model="SVM",
    accuracy=(0.769, 0.763, 0.8129496402877698, 0.8237410071942446),
    f1=(0.772189349112426, 0.7662721893491123, 0.8522727272727274, 0.8528528528528528),
    recall=(0.783, 0.777, 0.872093023255814, 0.8255813953488372),
    precision=(0.7616731517509727, 0.7558365758754864, 0.8333333333333334, 0.8819875776397516),
)

DECISION_TREE = MetricCard(
    model="Decision tree",
    accuracy=(0.7265, 0.7295, 0.7769784172661871, 0.7985611510791367),
    f1=(0.7556945064761054, 0.7566351776878093, 0.8342245989304812, 0.8461538461538463),
    recall=(0.846, 0.841, 0.9069767441860465, 0.8953488372093024),
    precision=(0.6828087167070218, 0.687653311529027, 0.7722772277227723, 0.8020833333333334),
)

RANDOM_FOREST = MetricCard(
    model="Random forest",
    accuracy=(0.8145, 0.818, 0.8057553956834532, 0.8525179856115108),
    f1=(0.8232491662696524, 0.8255033557046979, 0.8524590163934426, 0.8838526912181304),
    recall=(0.864, 0.861, 0.9069767441860465, 0.9069767441860465),
    precision=(0.7861692447679709, 0.7928176795580111, 0.8041237113402062, 0.861878453038674),
)

GRADIENT_BOOSTING = MetricCard(
    model="Gradient boosting",
    accuracy=(0.746, 0.7475, 0.7949640287769785, 0.7949640287769785),
    f1=(0.75146771037182, 0.7528144884973079, 0.8376068376068376, 0.8357348703170029),
)

BI_LSTM = MetricCard(
    model="Bi-LSTM + attention",
    accuracy=(0.8635, 0.8735, 0.8669064748201439, 0.8920863309352518),
    f1=(0.8655834564254061, 0.8685714285714285, 0.8939828080229226, 0.9107142857142858),
    recall=(0.879, 0.836, 0.9069767441860465, 0.8895348837209303),
    precision=(0.8525703200775946, 0.9037837837837838, 0.8813559322033898, 0.9329268292682927),
)

ALL_CARDS: Tuple[MetricCard, ...] = (
    DECISION_TREE,
    SVM,
    GRADIENT_BOOSTING,
    RANDOM_FOREST,
    BI_LSTM,
)

HEADLINE_CARDS: Tuple[MetricCard, ...] = (
    DECISION_TREE,
    SVM,
    RANDOM_FOREST,
    BI_LSTM,
)


def markdown_table(cards: Sequence[MetricCard], metric: str) -> str:
    header = (
        f"| Model | {CONDITION_LABELS['test_word']} | "
        f"{CONDITION_LABELS['test_word_emoji']} | "
        f"{CONDITION_LABELS['subtest_word']} | "
        f"{CONDITION_LABELS['subtest_word_emoji']} |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: |"
    rows = [header, sep]
    for card in cards:
        cells = card.percent(metric)
        rows.append(
            f"| {card.model} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"
        )
    return "\n".join(rows)


def ascii_table(cards: Sequence[MetricCard], metric: str) -> str:
    headers = ["Model", *(CONDITION_LABELS[name] for name in CONDITIONS)]
    body: List[List[str]] = []
    for card in cards:
        body.append([card.model, *card.percent(metric)])
    widths = [len(col) for col in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in body)
    return "\n".join(lines)


def emoji_gain(card: MetricCard, metric: str = "accuracy") -> Dict[str, float]:
    values = card.as_row(metric)
    return {
        "test_gain": values[1] - values[0],
        "subtest_gain": values[3] - values[2],
    }


def best_card(cards: Iterable[MetricCard], metric: str, condition: int) -> MetricCard:
    return max(cards, key=lambda card: card.as_row(metric)[condition])
