"""Published 2023 notebook numbers, stored as data instead of screenshots.

Values come from ``get_metrics_of_models.ipynb``. They are the personal
project's recorded test / emoji-subtest scores, not a live evaluation of the
incomplete SavedModel folders in ``model/``.
"""

from __future__ import annotations

from dataclasses import dataclass


CONDITIONS = (
    "single_modal_test",
    "multi_modal_test",
    "single_modal_subtest",
    "multi_modal_subtest",
)


@dataclass(frozen=True)
class ScoreRow:
    model: str
    metric: str
    values: tuple[float, float, float, float]

    def as_percent(self) -> tuple[str, ...]:
        return tuple(f"{100 * value:.2f}" for value in self.values)


# Accuracy, F1, recall, precision in the four conditions above.
RECORDED_SCORES: tuple[ScoreRow, ...] = (
    ScoreRow("SVM", "accuracy", (0.7690, 0.7630, 0.8129, 0.8237)),
    ScoreRow("SVM", "f1", (0.7722, 0.7663, 0.8523, 0.8529)),
    ScoreRow("SVM", "recall", (0.7830, 0.7770, 0.8721, 0.8256)),
    ScoreRow("SVM", "precision", (0.7617, 0.7558, 0.8333, 0.8820)),
    ScoreRow("Decision Tree", "accuracy", (0.7265, 0.7295, 0.7770, 0.7986)),
    ScoreRow("Decision Tree", "f1", (0.7557, 0.7566, 0.8342, 0.8462)),
    ScoreRow("Decision Tree", "recall", (0.8460, 0.8410, 0.9070, 0.8953)),
    ScoreRow("Decision Tree", "precision", (0.6828, 0.6877, 0.7723, 0.8021)),
    ScoreRow("Random Forest", "accuracy", (0.8145, 0.8180, 0.8058, 0.8525)),
    ScoreRow("Random Forest", "f1", (0.8232, 0.8255, 0.8525, 0.8839)),
    ScoreRow("Random Forest", "recall", (0.8640, 0.8610, 0.9070, 0.9070)),
    ScoreRow("Random Forest", "precision", (0.7862, 0.7928, 0.8041, 0.8619)),
    ScoreRow("Gradient Boosting", "accuracy", (0.7460, 0.7475, 0.7950, 0.7950)),
    ScoreRow("Gradient Boosting", "f1", (0.7515, 0.7528, 0.8376, 0.8357)),
    ScoreRow("Bi-LSTM + Attention", "accuracy", (0.8635, 0.8735, 0.8669, 0.8921)),
    ScoreRow("Bi-LSTM + Attention", "f1", (0.8656, 0.8686, 0.8940, 0.9107)),
    ScoreRow("Bi-LSTM + Attention", "recall", (0.8790, 0.8360, 0.9070, 0.8895)),
    ScoreRow("Bi-LSTM + Attention", "precision", (0.8526, 0.9038, 0.8814, 0.9329)),
)


def rows_for_metric(metric: str) -> list[ScoreRow]:
    return [row for row in RECORDED_SCORES if row.metric == metric]


def markdown_table(metric: str = "accuracy") -> str:
    rows = rows_for_metric(metric)
    header = (
        f"| Model | Single-modal test | Multi-modal test | "
        f"Single-modal subtest | Multi-modal subtest |\n"
        f"| --- | ---: | ---: | ---: | ---: |"
    )
    body = []
    for row in rows:
        cells = " | ".join(row.as_percent())
        body.append(f"| {row.model} | {cells} |")
    return "\n".join([header, *body])
