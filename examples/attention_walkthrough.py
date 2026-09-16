"""Print Raffel attention weights on a synthetic sarcastic sequence."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .attention_numpy import (
    demo_sequence,
    demo_weight,
    raffel_attention,
    uniform_pool,
)


TOKENS = ("i", "love", "monday", "mornings", "#not")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-last",
        action="store_true",
        help="drop the #not timestep to show the mask path",
    )
    return parser


def _print_weights(title: str, weights: np.ndarray) -> None:
    print(title)
    for token, weight in zip(TOKENS, weights.reshape(-1)):
        bar = "#" * int(round(float(weight) * 40))
        print(f"  {token:<10} {weight:6.3f}  {bar}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    x = demo_sequence()
    w = demo_weight()
    mask = None
    if args.mask_last:
        mask = np.array([1, 1, 1, 1, 0], dtype=np.float64)

    result = raffel_attention(x, w, mask=mask)
    pooled = uniform_pool(x, mask=mask)

    print("Hidden size 4. The last unit is a synthetic sarcasm dimension.")
    print("W is biased toward that unit, so #not should dominate α.")
    print()
    _print_weights("attention weights α", result.weights)
    print()
    print("context vector     ", np.round(result.context.reshape(-1), 3))
    print("uniform mean pool  ", np.round(pooled.reshape(-1), 3))
    print()
    print(
        "The context is closer to the #not row than the mean is, unless "
        "--mask-last hides that step."
    )
    if mask is None:
        last = x[-1]
        print("raw #not row       ", np.round(last, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
