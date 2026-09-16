"""Pretty-print helpers for the 2023 reported score table."""

from __future__ import annotations

from .results import iter_reported_rows, load_reported


def format_reported_table() -> str:
    rows = iter_reported_rows()
    header = f"{'model':20s} {'split':8s} {'mode':8s} {'acc':8s} {'f1':8s}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['model']:20s} {row['split']:8s} {row['mode']:8s} "
            f"{row['acc']:.4f}   {row['f1']:.4f}"
        )
    source = load_reported().get("source", "")
    if source:
        lines.append("")
        lines.append(f"source: {source}")
    return "\n".join(lines)
