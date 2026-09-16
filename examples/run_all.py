"""Run every walkthrough script in a fixed order."""

from __future__ import annotations

import sys

from . import (
    attention_walkthrough,
    embedding_matrix_walkthrough,
    inspect_dataset,
    lexical_baseline,
    multimodal_fusion,
    reported_results,
    tokenize_tweets,
)

STEPS = (
    ("inspect_dataset", inspect_dataset.main, ["--examples", "2"]),
    ("reported_results", reported_results.main, ["--metric", "accuracy"]),
    ("tokenize_tweets", tokenize_tweets.main, ["--split", "subtest", "--limit", "3"]),
    ("attention_walkthrough", attention_walkthrough.main, []),
    ("multimodal_fusion", multimodal_fusion.main, []),
    ("embedding_matrix_walkthrough", embedding_matrix_walkthrough.main, []),
    ("lexical_baseline", lexical_baseline.main, ["--max-train", "4000", "--epochs", "80"]),
)


def main(argv: list[str] | None = None) -> int:
    del argv
    for name, func, args in STEPS:
        print("=" * 72)
        print(name)
        print("=" * 72)
        code = func(args)
        if code:
            print(f"{name} exited {code}", file=sys.stderr)
            return code
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
