#!/usr/bin/env python3
"""NumPy clone of attention_layer.Attention, with toy sarcasm sequences.

The Keras layer does:

    e_t = tanh(x_t · W + b_t)
    α   = softmax(e)          # + epsilon, optional mask
    out = Σ_t α_t x_t

This script uses those equations on tiny hand-built sequences so you can
see a downshift token (😒, #not) take more mass than a positive predicate
when W is aligned with a "clash" direction. It does not load TensorFlow
or the saved course models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cosine  # noqa: E402

EPS = np.finfo(np.float32).eps


def attention_scores(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(context, alpha, eij)`` for a single sequence ``[T, D]``."""
    if x.ndim != 2:
        raise ValueError(f"expected [timesteps, features], got {x.shape}")
    eij = x.dot(w)
    if b is not None:
        if b.shape[0] != x.shape[0]:
            raise ValueError(
                f"bias length {b.shape[0]} != timesteps {x.shape[0]}. "
                "This is the same constraint as Attention.build."
            )
        eij = eij + b
    eij = np.tanh(eij)
    unnormalized = np.exp(eij)
    if mask is not None:
        unnormalized = unnormalized * mask.astype(unnormalized.dtype)
    alpha = unnormalized / (unnormalized.sum() + EPS)
    context = (x * alpha[:, None]).sum(axis=0)
    return context, alpha, eij


def softmax_entropy(alpha: np.ndarray) -> float:
    clipped = np.clip(alpha, EPS, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def make_basis(dim: int, seed: int = 0) -> dict:
    """Orthonormal-ish directions: positive affect, downshift, padding, other."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(4, dim))
    # QR gives orthonormal rows after we take Q.T? Use QR on columns.
    q, _ = np.linalg.qr(raw.T)
    axes = q.T  # 4 x dim, orthonormal
    return {
        "positive": axes[0],
        "downshift": axes[1],
        "padding": axes[2],
        "other": axes[3],
    }


TOKEN_AXIS = {
    "love": "positive",
    "great": "positive",
    "best": "positive",
    "unamused": "downshift",
    "not": "downshift",
    "dirty": "downshift",
    "pad": "padding",
    "the": "other",
    "when": "other",
}


def embed_tokens(tokens: list[str], axes: dict, dim: int) -> np.ndarray:
    rows = []
    for tok in tokens:
        axis = TOKEN_AXIS.get(tok)
        if axis is None:
            raise KeyError(f"unknown toy token {tok!r}")
        rows.append(axes[axis])
    return np.stack(rows, axis=0)


def show_sequence(name: str, tokens: list[str], w: np.ndarray, axes: dict, dim: int) -> None:
    x = embed_tokens(tokens, axes, dim)
    mask = np.array([0.0 if tok == "pad" else 1.0 for tok in tokens])
    context, alpha, eij = attention_scores(x, w, b=None, mask=mask)
    print(f"--- {name} ---")
    print(f"tokens:  {tokens}")
    print(" token        e_t      alpha")
    for tok, e, a in zip(tokens, eij, alpha):
        bar = "#" * int(round(30 * float(a)))
        print(f"  {tok:10s}  {e:+6.3f}   {a:6.3f}  {bar}")
    print(
        f"entropy(alpha)={softmax_entropy(alpha):.3f}  "
        f"cos(context, downshift)={cosine(context, axes['downshift']):+.3f}  "
        f"cos(context, positive)={cosine(context, axes['positive']):+.3f}"
    )
    print()


def assert_invariants(dim: int = 16) -> None:
    """A few numerical checks that match attention_layer.py."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, dim))
    w = rng.normal(size=(dim,))
    b = rng.normal(size=(5,))
    context, alpha, _ = attention_scores(x, w, b=b, mask=None)
    if not np.isclose(alpha.sum(), 1.0, atol=1e-6):
        raise AssertionError(f"alpha must sum to 1, got {alpha.sum()}")
    # Masking the last two steps should put ~0 mass there.
    mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    _, alpha_m, _ = attention_scores(x, w, b=b, mask=mask)
    if alpha_m[3] > 1e-8 or alpha_m[4] > 1e-8:
        raise AssertionError(f"masked steps still have mass: {alpha_m}")
    if not np.isclose(alpha_m.sum(), 1.0, atol=1e-6):
        raise AssertionError("masked softmax must still normalize")
    # Uniform W=0, b=0 → tanh(0)=0 → uniform alpha over unmasked steps.
    _, alpha_u, _ = attention_scores(x, np.zeros(dim), b=np.zeros(5), mask=None)
    expected = np.full(5, 1.0 / 5.0)
    if not np.allclose(alpha_u, expected, atol=1e-6):
        raise AssertionError(f"zero scores should be uniform, got {alpha_u}")
    # Output dim is features, not timesteps (compute_output_shape).
    if context.shape != (dim,):
        raise AssertionError(f"context shape {context.shape} != ({dim},)")
    print("invariants: ok (softmax, mask, uniform-zero, output rank)\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=32, help="Toy embedding size.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-invariants",
        action="store_true",
        help="Do not run the numerical self-check.",
    )
    args = parser.parse_args(argv)

    if not args.skip_invariants:
        assert_invariants(dim=max(8, args.dim))

    axes = make_basis(args.dim, seed=args.seed)
    # W points at the downshift axis: the layer prefers clash / negation tokens.
    w = axes["downshift"]

    print(
        "W is aligned with the toy 'downshift' axis. "
        "Attention should pile mass on 😒 / #not / dirty, not on love/great.\n"
    )
    show_sequence(
        "positive wording + unamused emoji",
        ["love", "the", "unamused"],
        w,
        axes,
        args.dim,
    )
    show_sequence(
        "love ... dirty #not, then padding",
        ["love", "when", "dirty", "not", "pad", "pad"],
        w,
        axes,
        args.dim,
    )
    show_sequence(
        "sincere positive, no downshift",
        ["great", "the", "best"],
        w,
        axes,
        args.dim,
    )
    print(
        "Takeaway: the Keras layer is a masked tanh-attention. "
        "It can surface a local sarcasm clash, but only if the hidden "
        "state of that token actually points near W. The real network "
        "learns W; this demo just aims it at the downshift axis."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
