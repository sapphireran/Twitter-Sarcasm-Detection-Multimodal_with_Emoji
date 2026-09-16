"""Plain-text tables for the example scripts."""

from __future__ import annotations


def format_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> str:
    right_align = right_align or set()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_cell(i: int, cell: str) -> str:
        return cell.rjust(widths[i]) if i in right_align else cell.ljust(widths[i])

    header = "  ".join(fmt_cell(i, h) for i, h in enumerate(headers))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(fmt_cell(i, cell) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([header, rule, *body])
