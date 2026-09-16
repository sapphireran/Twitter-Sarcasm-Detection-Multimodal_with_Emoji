#!/usr/bin/env python3
"""Run every walkthrough script and fail on the first non-zero exit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ["python", "examples/inspect_dataset.py"],
    ["python", "examples/preprocess_walkthrough.py", "--n", "4"],
    ["python", "examples/emoji_signal.py"],
    ["python", "examples/heuristic_baseline.py"],
    ["python", "examples/attention_demo.py"],
    ["python", "examples/report_metrics.py"],
]


def main() -> int:
    for command in SCRIPTS:
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            print(f"FAILED {' '.join(command)} (exit {completed.returncode})", file=sys.stderr)
            return completed.returncode
        print()
    print("all example scripts exited 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
