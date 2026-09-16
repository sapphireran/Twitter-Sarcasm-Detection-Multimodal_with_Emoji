#!/usr/bin/env python3
"""Walk through the Raffel attention equations used in attention_layer.py."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np

from sarcasm_lab.attention import (
    attention_entropy,
    expected_step,
    make_demo_sequence,
    raffel_attention,
    softmax,
)


def _bar(weight: float, width: int = 24) -> str:
    n = int(round(weight * width))
    return "█" * n + "·" * (width - n)


def main() -> int:
    print("Keras layer (attention_layer.py)")
    print("  e_t = tanh(x_t · W + b_t)")
    print("  a   = softmax(e)   # mask applied after exp, + epsilon in the denom")
    print("  h   = sum_t a_t x_t")
    print()
    print("The Keras bias is length `steps`, so padded length is part of the layer.")
    print("Saved models in this repo used maxlen=78.")
    print()

    demo = make_demo_sequence()
    attn = demo["attention"][0]
    mask = demo["mask"][0]
    print("Synthetic tweet: bland positive opener, two padded slots, late reversal cue")
    print(f"{'step':>4}  {'mask':>4}  {'a_t':>8}  attention")
    labels = [
        "I",
        "love",
        "walking",
        "to",
        "<pad>",
        "<pad>",
        "school",
        "#not",
    ]
    for i, (weight, keep, label) in enumerate(zip(attn, mask, labels)):
        marker = "" if keep else "  (masked)"
        print(f"{i:4d}  {int(keep):4d}  {weight:8.4f}  {_bar(float(weight))}  {label}{marker}")

    print()
    print(f"sum(a)             = {attn.sum():.6f}  (must be 1)")
    print(f"entropy            = {float(attention_entropy(demo['attention'])[0]):.3f}")
    print(f"expected step      = {float(expected_step(demo['attention'])[0]):.2f}  (cue is at step 7)")
    print(f"context vector     = {np.round(demo['context'][0], 3)}")

    print()
    print("Same sequence without the mask — pad slots can steal a little mass")
    context_u, attn_u = raffel_attention(demo["x"], demo["weight"], bias=demo["bias"], mask=None)
    print(f"unmasked a[4], a[5] = {attn_u[0, 4]:.4f}, {attn_u[0, 5]:.4f}")
    print(f"masked   a[4], a[5] = {attn[4]:.4f}, {attn[5]:.4f}")
    print(f"L2(context change)  = {float(np.linalg.norm(context_u - demo['context'])):.4f}")

    print()
    print("Softmax numerical guard (matches the Keras epsilon trick)")
    tiny = np.array([[-50.0, -50.0, -50.0]])
    weights = softmax(tiny)
    print(f"softmax([-50,-50,-50]) = {weights[0]}  sum={weights[0].sum():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
