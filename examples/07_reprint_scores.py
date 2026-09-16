#!/usr/bin/env python3
"""Reprint the June 2023 notebook scores as a markdown table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccs2lab.recorded import markdown_table


def main() -> int:
    print("# Recorded 2023 scores")
    print()
    print(
        "Source: `get_metrics_of_models.ipynb` and "
        "`evaluate_loaded_dl_models.ipynb`. W = word only, WE = word + emoji."
    )
    print()
    print(markdown_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
