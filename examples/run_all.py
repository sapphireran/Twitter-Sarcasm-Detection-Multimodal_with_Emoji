#!/usr/bin/env python3
"""Run every example script and fail if any of them exits non-zero."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "examples/dataset_overview.py",
    "examples/tokenize_demo.py",
    "examples/embedding_demo.py",
    "examples/attention_demo.py",
    "examples/sarcasm_cues.py",
    "examples/toy_pipeline.py",
]


def main() -> int:
    failed = 0
    for script in SCRIPTS:
        print("=" * 72)
        print(script)
        print("=" * 72)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            failed += 1
            print(f"{script} failed with {result.returncode}", file=sys.stderr)
        print()
    if failed:
        print(f"{failed} example script(s) failed")
        return 1
    print("all example scripts exited 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
