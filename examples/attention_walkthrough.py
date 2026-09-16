#!/usr/bin/env python3
"""Score a synthetic tweet with the Raffel-style attention equations.

The 2023 Keras layer lives in ``attention_layer.py``. This script uses the
stdlib clone in ``examples/lib/attention.py`` so you can see which token would
be weighted most heavily for a given score vector ``W`` — without TensorFlow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.attention import argmax_token, temporal_attention  # noqa: E402
from lib.tokenize import tokenize_tweet  # noqa: E402


def _one_hotish(tokens, lexicon):
    """Map each token to a small hand-built hidden state.

    Dims: [positive, negative-situation, leak-hashtag, emoji-flat].
    This is *not* a trained BiLSTM; it is a cartoon of the directions attention
    could lock onto.
    """
    rows = []
    for tok in tokens:
        row = [0.05, 0.05, 0.05, 0.05]
        if tok in {"love", "great", "awesome", "yay"}:
            row[0] = 1.2
        if tok in {"wait", "waiting", "late", "work", "hair", "shift"}:
            row[1] = 1.0
        if tok.startswith("#") and tok in {"#not", "#sarcasm", "#yeahright"}:
            row[2] = 1.4
        if len(tok) == 1 and ord(tok) > 255:
            row[3] = 1.1
        # unknown tokens stay near zero so the pad-like prior does not dominate
        if tok in lexicon:
            pass
        rows.append(row)
    return rows


def render(text: str, weight, bias=None) -> None:
    tokens = tokenize_tweet(text)
    hidden = _one_hotish(tokens, set(tokens))
    pooled, alphas, scores = temporal_attention(hidden, weight, bias=bias)
    width = max(len(tok) for tok in tokens) if tokens else 1
    print(f"tweet: {text}")
    print(f"{'token':<{width}}  score    alpha   bar")
    for tok, score, alpha in zip(tokens, scores, alphas):
        bar = "#" * int(round(alpha * 40))
        print(f"{tok:<{width}}  {score:+.3f}   {alpha:0.3f}   {bar}")
    print(f"attended vector: {[round(x, 3) for x in pooled]}")
    print(f"argmax token:    {argmax_token(tokens, alphas)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefer",
        choices=("leak", "emoji", "positive"),
        default="leak",
        help="which hidden dimension W should lock onto",
    )
    parser.add_argument("--text", help="optional extra tweet")
    args = parser.parse_args()

    prefer = {
        "positive": [1.0, 0.0, 0.0, 0.0],
        "emoji": [0.0, 0.0, 0.0, 1.0],
        "leak": [0.0, 0.0, 1.0, 0.0],
    }[args.prefer]

    print(f"W prefers the {args.prefer!r} dimension: {prefer}\n")
    tweets = [
        "I just love having grungy ass hair 😑 #not",
        "3 more days until I'm reunited with my friends yay 😍",
        "Don't you love waiting in line",
    ]
    if args.text:
        tweets.append(args.text)
    for tweet in tweets:
        render(tweet, prefer)


if __name__ == "__main__":
    main()
