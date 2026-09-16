#!/usr/bin/env python3
"""Walk a 3-step tweet through the Raffel attention used in attention_layer.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.attention_numpy import attention_forward

# One tweet: "love    #not    <pad>"
# feature 0 = positive-word, feature 1 = negation-hashtag, feature 2 = pad flag
X = np.asarray(
    [
        [
            [0.9, 0.0, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ],
    dtype=np.float64,
)

# W leans on the negation channel so #not should win the softmax.
W = np.asarray([0.2, 1.4, -0.8], dtype=np.float64)
B = np.asarray([0.0, 0.1, 0.0], dtype=np.float64)
MASK = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float64)


def _print(name: str, context: np.ndarray, alpha: np.ndarray) -> None:
    print(name)
    print("  alpha   ", np.round(alpha[0], 4))
    print("  context ", np.round(context[0], 4))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    print("steps: [love, #not, <pad>]")
    print("W     :", W)
    print()

    ctx_unmasked, a_unmasked = attention_forward(X, W, B, mask=None)
    ctx_masked, a_masked = attention_forward(X, W, B, mask=MASK)

    _print("no mask (pad is a real step — this is what PrepModel does)", ctx_unmasked, a_unmasked)
    _print("with mask (pad zeroed after exp, then renormalized)", ctx_masked, a_masked)

    if a_masked[0, 1] <= a_masked[0, 0]:
        raise SystemExit("expected #not to outrank love under this W")
    if a_masked[0, 2] > 1e-12:
        raise SystemExit("masked pad should have ~0 mass")

    print(
        "PrepModel never sets mask_zero=True, so the shipped graph is closer "
        "to the first block. A 78-step pad can still steal a little α."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
