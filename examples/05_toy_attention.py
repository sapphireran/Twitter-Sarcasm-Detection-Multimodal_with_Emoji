#!/usr/bin/env python3
"""Walk through Raffel-style attention on a tiny hidden-state sequence.

The Keras layer in ``attention_layer.py`` is hard to inspect once it is baked
into a SavedModel. This script:

* builds a 1 × T × D hidden tensor that peaks on one timestep
* runs the same energy / softmax / weighted-sum steps as the layer
* shows that the context vector moves toward the attended hidden state
* demonstrates the mask path used for padded tweets

Run::

    python3 examples/05_toy_attention.py
    python3 examples/05_toy_attention.py --steps 6 --features 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.common.attention import AdditiveAttention


def explain(steps: int, features: int, peak: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    # Construct hidden states that are mostly noise, except index `peak`
    # which is a scaled copy of W so that e_peak = tanh(||W||^2) is large.
    w = rng.normal(0.0, 0.4, size=(features,)).astype(np.float32)
    hidden = rng.normal(0.0, 0.05, size=(1, steps, features)).astype(np.float32)
    hidden[0, peak] = w * 3.0
    attn = AdditiveAttention(w=w, bias=np.zeros((steps,), dtype=np.float32))
    energy = attn.scores(hidden)[0]
    weights = attn.weights(hidden)[0]
    context = attn.context(hidden)[0]
    # Cosine between context and each hidden state.
    def _cos(a, b):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    lines = [
        f"hidden shape = (1, {steps}, {features})",
        f"planted peak at timestep t={peak}",
        "",
        f"{'t':>4} {'energy':>10} {'alpha':>10} {'cos(ctx, h_t)':>16}",
        "-" * 44,
    ]
    for t in range(steps):
        lines.append(
            f"{t:4d} {energy[t]:10.4f} {weights[t]:10.4f} {_cos(context, hidden[0, t]):16.4f}"
        )
    lines.append("")
    lines.append(f"sum(alpha) = {weights.sum():.6f}  (should be 1)")
    lines.append(
        f"argmax(alpha) = {int(weights.argmax())}  "
        f"(should match planted peak {peak})"
    )
    lines.append("")
    lines.append("Padding mask")
    lines.append("------------")
    mask = np.ones((1, steps), dtype=np.float32)
    # Pretend the last two tokens are pad.
    if steps >= 3:
        mask[0, -2:] = 0.0
        w_masked = attn.weights(hidden, mask=mask)[0]
        lines.append(
            "mask = " + " ".join(str(int(x)) for x in mask[0])
        )
        lines.append(
            "alpha_masked = " + " ".join(f"{a:.3f}" for a in w_masked)
        )
        lines.append(
            f"mass on padded steps = {w_masked[-2:].sum():.6f}  (should be ~0)"
        )
    lines.append("")
    lines.append("This is the mechanism sitting on top of the two BiLSTM layers")
    lines.append("in dl_model.PrepModel. After training, alpha typically spikes on")
    lines.append("cue hashtags, sentiment flips (\"love\" in a negative tweet), and")
    lines.append("emoji tokens when the embedding row is non-zero.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--peak", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not (0 <= args.peak < args.steps):
        parser.error("--peak must be in [0, steps)")
    print(explain(args.steps, args.features, args.peak, args.seed), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
