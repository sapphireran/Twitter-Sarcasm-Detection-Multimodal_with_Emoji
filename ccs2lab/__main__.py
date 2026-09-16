"""Run the archive lab suite: ``python -m ccs2lab``."""

from __future__ import annotations

import argparse
import runpy
import sys

from ccs2lab.paths import ROOT

SCRIPTS = (
    "examples/01_split_census.py",
    "examples/02_cue_shift.py",
    "examples/03_emoji2vec_probe.py",
    "examples/04_attention_replay.py",
    "examples/05_cue_rule.py",
    "examples/06_hashed_baseline.py",
    "examples/07_reprint_scores.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CCS2 2023 archive lab.")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the hashed logistic fit (faster smoke run).",
    )
    args = parser.parse_args(argv)
    scripts = list(SCRIPTS)
    if args.skip_baseline:
        scripts = [path for path in scripts if "06_hashed" not in path]
    for rel in scripts:
        path = ROOT / rel
        print(f"\n===== {rel} =====\n", flush=True)
        sys.argv = [str(path)]
        runpy.run_path(str(path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
