#!/usr/bin/env python3
"""Smoke-run every example script with non-interactive defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent

JOBS = (
    ["explore_dataset.py", "--split", "test"],
    ["tokenize_tweets.py", "--synthetic-only"],
    ["inspect_emoji2vec.py", "--neighbors", "😒", "😍"],
    ["attention_demo.py"],
    ["cue_baseline.py", "--split", "test"],
    ["reprint_course_results.py", "--metric", "acc"],
)


def main() -> int:
    failed = 0
    for job in JOBS:
        script = EXAMPLES / job[0]
        cmd = [sys.executable, str(script), *job[1:]]
        print(f"\n########## {' '.join(job)} ##########\n", flush=True)
        completed = subprocess.run(cmd, cwd=str(ROOT))
        if completed.returncode != 0:
            print(f"FAILED {job[0]} with {completed.returncode}", file=sys.stderr)
            failed += 1
    if failed:
        print(f"\n{failed} example(s) failed", file=sys.stderr)
        return 1
    print("\nall examples ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
