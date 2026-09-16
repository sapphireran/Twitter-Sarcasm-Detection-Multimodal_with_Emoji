#!/usr/bin/env python3
"""Step through Raffel attention on a 5-token sarcastic tweet.

The Keras layer is `attention_layer.Attention`. This script uses the
NumPy twin in `examples/lib/attention_numpy.py` so you can see the
energies, the softmax, and the weighted sum without TensorFlow.

Usage:

    python3 examples/attention_walkthrough.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.attention_numpy import demo_sequence, raffel_attention  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-last",
        action="store_true",
        help="zero the #not step to show the mask path (alpha re-normalizes)",
    )
    args = parser.parse_args(argv)

    demo = demo_sequence()
    tokens = list(demo["tokens"])
    x = demo["x"]
    weight = demo["weight"]
    bias = demo["bias"]
    mask = demo["mask"].copy()
    if args.mask_last:
        mask[0, -1] = 0.0

    context, alpha = raffel_attention(x, weight, bias=bias, mask=mask)

    print("tweet steps:", " / ".join(tokens))
    print("feature meaning: [sarcasm_chan, positive_chan, extra, extra]")
    print()
    print(f"{'token':<16}{'e=x·W+b':>10}{'tanh(e)':>10}{'α':>10}{'masked':>8}")
    energy = np.tensordot(x, weight, axes=([-1], [0])) + bias
    tanh_e = np.tanh(energy)
    for i, tok in enumerate(tokens):
        print(
            f"{tok:<16}{energy[0, i]:10.3f}{tanh_e[0, i]:10.3f}"
            f"{alpha[0, i]:10.3f}{mask[0, i]:8.0f}"
        )
    print()
    print("context (weighted sum of steps):")
    print("  " + " ".join(f"{v:+.3f}" for v in context[0]))
    print()
    print(
        f"mass on 😒 + #not: {alpha[0, 3] + alpha[0, 4]:.3f}   "
        f"mass on 'love'+'i'+'this': {alpha[0, 0] + alpha[0, 1] + alpha[0, 2]:.3f}"
    )
    if args.mask_last:
        print("last step was masked; softmax mass moved onto the remaining tokens.")
    else:
        print(
            "W looks at the sarcasm channel, so α should peak on 😒 and #not "
            "— the same late-cue pattern the coursework LSTM is built to catch."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
