#!/usr/bin/env python3
"""Run every example script in a stable order."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.inspect_dataset import main as inspect_main
from examples.run_attention import main as attention_main
from examples.run_pipeline import main as pipeline_main
from examples.run_real_tokens import main as real_tokens_main
from examples.run_toy_classifier import main as classifier_main

STEPS = (
    ("examples/inspect_dataset.py", inspect_main),
    ("examples/run_real_tokens.py", real_tokens_main),
    ("examples/run_pipeline.py", pipeline_main),
    ("examples/run_attention.py", attention_main),
    ("examples/run_toy_classifier.py", classifier_main),
)


def main() -> int:
    print("Running personal sarcasm-detection examples\n")
    for rel, fn in STEPS:
        print("=" * 72)
        print(rel)
        print("=" * 72)
        code = fn()
        if code:
            return int(code)
        print()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
