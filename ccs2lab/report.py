"""Shared formatting for the archive lab CLIs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    align = []
    for title in headers:
        align.append("---:" if title.lower() in {"n", "acc", "accuracy", "f1", "precision", "recall", "coverage", "p", "lo", "hi", "or"} or title.endswith("%") else "---")
    sep = "| " + " | ".join(align) + " |"
    body = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, float):
                cells.append(f"{cell:.4f}")
            else:
                cells.append(str(cell))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"
