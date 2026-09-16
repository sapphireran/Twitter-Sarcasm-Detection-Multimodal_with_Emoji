"""``python -m sarcasm_toolkit`` prints split sizes and the reported table."""

from __future__ import annotations

from .dataset import load_split, summarize_split
from .reported import format_reported_table


def main() -> None:
    print("Local dataset splits")
    print("====================")
    for name in ("train", "test", "subtest"):
        summary = summarize_split(load_split(name))
        print(
            f"{summary['split']:8s}  n={summary['n']:5d}  "
            f"sarcastic={summary['sarcastic']:5d} "
            f"({summary['sarcastic_rate']:.3f})  "
            f"mean tokens={summary['token_len']['mean']:.1f}"
        )
    print()
    print("Notebook-reported scores (2023)")
    print("===============================")
    print(format_reported_table())


if __name__ == "__main__":
    main()
