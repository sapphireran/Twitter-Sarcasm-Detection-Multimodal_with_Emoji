#!/usr/bin/env python3
"""Run every walkthrough script and fail on the first non-zero exit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPTS = [
    [PY, "examples/inspect_dataset.py"],
    [PY, "examples/preprocess_walkthrough.py", "--n", "4"],
    [PY, "examples/emoji_signal.py"],
    [PY, "examples/heuristic_baseline.py"],
    [PY, "examples/attention_demo.py"],
    [PY, "examples/report_metrics.py"],
]


def main() -> int:
    for command in SCRIPTS:
        printable = ["python" if part == PY else part for part in command]
        print("+", " ".join(printable), flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            print(f"FAILED {' '.join(command)} (exit {completed.returncode})", file=sys.stderr)
            return completed.returncode
        print()
    print("all example scripts exited 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
