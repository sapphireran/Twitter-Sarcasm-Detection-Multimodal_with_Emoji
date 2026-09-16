#!/usr/bin/env python3
"""Raffel-style attention pooling on toy token vectors.

The Keras ``Attention`` layer in ``attention_layer.py`` scores each
timestep with ``tanh(x · W + b)``, softmaxes, and returns the weighted
sum. This example uses the same math on 8-d toy embeddings so you can
see the cue hashtag and emoji attract mass without loading GloVe.
"""

from __future__ import annotations

import math

import _path  # noqa: F401

from sarcasm_toolkit.attention import attention_pool, attention_report
from sarcasm_toolkit.embeddings import TOY_DIM, average_pool, embed_tokens
from sarcasm_toolkit.tokenize import tokenize_tweet

# W emphasizes sarcasm-cue (dim 2) and emoji/affect (dim 3) — the axes
# that ``embeddings.SEED_VECTORS`` reserved for #not / 😒.
ATTENTION_W = [0.05, 0.05, 1.40, 1.10, 0.15, 0.05, 0.05, 0.05]


DEMO_TWEETS = [
    "I just love having grungy ass hair 😑 #not",
    "I loovee when people text back ... 😒 #sarcastictweet",
    "<user> i hope youre lurking rn pretty please?! 😭 😭 😭",
    "Being sore is the best and the worst feeling in the world",
]


def cosine(left: list[float], right: list[float]) -> float:
    num = sum(a * b for a, b in zip(left, right))
    den_l = math.sqrt(sum(a * a for a in left))
    den_r = math.sqrt(sum(b * b for b in right))
    if den_l == 0 or den_r == 0:
        return 0.0
    return num / (den_l * den_r)


def show_tweet(text: str) -> None:
    tokens = tokenize_tweet(text)
    vectors = embed_tokens(tokens, dim=TOY_DIM)
    bias = [0.0] * len(tokens)
    weights = attention_report(vectors, ATTENTION_W, tokens=tokens, bias=bias)
    pooled = attention_pool([vectors], ATTENTION_W, bias=bias)[0]
    mean = average_pool(vectors, dim=TOY_DIM)
    print(f"\n{text}")
    print(f"  tokens: {tokens}")
    ranked = sorted(weights, key=lambda row: row["weight"], reverse=True)
    print("  attention (high → low):")
    for row in ranked:
        bar = "#" * int(round(row["weight"] * 40))
        print(f"    {row['weight']:.3f}  {row['token']:18s} {bar}")
    print(f"  avg-pool vs attn-pool cosine: {cosine(mean, pooled):.3f}")
    print(
        "  avg-pool cue/emoji axes (2,3): "
        f"{mean[2]:.3f}, {mean[3]:.3f} | "
        f"attn-pool: {pooled[2]:.3f}, {pooled[3]:.3f}"
    )


def main() -> None:
    print("Attention pooling demo (pure Python, toy 8-d vectors)")
    print("W is biased toward the sarcasm-cue and emoji axes.")
    print("Compare with mean pooling: attention should upweight #not / emoji.")
    for tweet in DEMO_TWEETS:
        show_tweet(tweet)

    # Sanity: weights on one sequence sum to ~1.
    tokens = tokenize_tweet(DEMO_TWEETS[0])
    report = attention_report(embed_tokens(tokens), ATTENTION_W, tokens=tokens)
    total = sum(row["weight"] for row in report)
    print(f"\nchecksum: attention weights sum to {total:.6f} (expect ~1)")


if __name__ == "__main__":
    main()
