#!/usr/bin/env python3
"""Walk a tweet through the original preprocess stages without Keras.

The integer Tokenizer / pad_sequences / GloVe table from ``data_utils.py``
need packages that are not in this environment. This script shows the
stages that *can* be replayed: raw line → ReadOpen comma rewrite →
example tokenizer → embedding-row routing decision.

Usage:
    python3 examples/preprocess_walkthrough.py
    python3 examples/preprocess_walkthrough.py --n 6 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

from sarcasm_lib.dataset import (
    SPLIT_NAMES,
    faithful_readopen_text,
    load_split,
    strip_wrapping_quotes,
)
from sarcasm_lib.features import cue_flags
from sarcasm_lib.tokenize import extract_emoji, tokenize_tweet


def route_token(token: str) -> str:
    """Describe which embedding row ``Preprocess`` would try to write."""
    if token.startswith("#"):
        return "hashtag: GloVe if the full token (including '#') is in vocab, else zero"
    if token in {"<user>", "<url>"}:
        return "placeholder: GloVe if '<user>' / '<url>' was in the GloVe Twitter dump"
    if extract_emoji(token) == [token] or extract_emoji(token):
        return (
            "emoji: GloVe first; on miss, average emoji2vec rows when "
            "get_emoji2vec=True, else zeros (single-modal)"
        )
    if token.startswith("http"):
        return "url: almost always OOV → zero row"
    return "word: GloVe lookup; OOV → zero row"


def walk(raw_from_file: str, label: int) -> None:
    faithful = faithful_readopen_text(raw_from_file)
    preserved = strip_wrapping_quotes(raw_from_file.rstrip("\r\n"))
    tokens = tokenize_tweet(preserved)
    flags = cue_flags(preserved)
    print(f"label {label}")
    print(f"  file line:     {raw_from_file.rstrip()[:160]}")
    print(f"  preserved:     {preserved[:160]}")
    if faithful != preserved:
        print(f"  ReadOpen text: {faithful[:160]}")
    print(f"  tokens:        {tokens}")
    print(
        f"  cues:          emoji={flags.has_emoji}  #not={flags.has_not_hashtag}  "
        f"sarcasm_tag={flags.has_explicit_sarcasm_tag}  user={flags.has_user}"
    )
    print("  embedding routes:")
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        print(f"    {token!r:20} → {route_token(token)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=SPLIT_NAMES)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument(
        "--prefer-emoji",
        action="store_true",
        help="prefer tweets that contain emoji when sampling examples",
    )
    args = parser.parse_args(argv)

    split = load_split(args.split, faithful=False)
    # Re-read raw lines so we can show the file form including quotes.
    raw_path = ROOT / "dataset" / f"{args.split}_sentence.csv"
    raw_lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()

    print("# Preprocess walkthrough")
    print(
        "Stages shared with data_utils.ReadOpen / Preprocess, without "
        "building a Keras Tokenizer or a 200-d table."
    )
    print()
    print("Architecture reminder:")
    print("  tokens → frozen 200-d rows → Dropout → BiLSTM → Dropout →")
    print("  BiLSTM → Dropout → Attention → Dense(1, sigmoid)")
    print()

    picked: list[int] = []
    if args.prefer_emoji:
        for i, sentence in enumerate(split.sentences):
            if extract_emoji(sentence):
                picked.append(i)
            if len(picked) >= args.n:
                break
    if len(picked) < args.n:
        # Mix classes: first unused 0, first unused 1, then sequential.
        for label in (0, 1):
            for i, gold in enumerate(split.labels):
                if gold == label and i not in picked:
                    picked.append(i)
                    break
        i = 0
        while len(picked) < args.n and i < len(split):
            if i not in picked:
                picked.append(i)
            i += 1

    for index in picked[: args.n]:
        walk(raw_lines[index], split.labels[index])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
