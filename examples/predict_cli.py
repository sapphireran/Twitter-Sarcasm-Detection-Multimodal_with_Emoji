#!/usr/bin/env python3
"""Score one tweet, or a small batch file, with the explicit-cue heuristic.

This is the command-line companion to docs/results.md. It will not match the
Bi-LSTM. It will tell you whether a tweet is wearing its label on a hashtag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarcasm_lib.heuristic import score_tweet
from sarcasm_lib.tokenize import tokenize_tweet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="tweet text; omit to read stdin")
    parser.add_argument(
        "--file",
        type=Path,
        help="optional UTF-8 file with one tweet per line",
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser


def collect_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.append(" ".join(args.text))
    if args.file:
        texts.extend(
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not texts and not sys.stdin.isatty():
        texts.extend(line.strip() for line in sys.stdin if line.strip())
    return texts


def result_payload(text: str, threshold: float) -> dict[str, object]:
    scored = score_tweet(text, threshold=threshold)
    return {
        "text": scored.text,
        "tokens": tokenize_tweet(scored.text),
        "score": scored.score,
        "predicted": scored.predicted,
        "label": scored.label_name,
        "reasons": list(scored.reasons),
    }


def main() -> None:
    args = build_parser().parse_args()
    texts = collect_texts(args)
    if not texts:
        raise SystemExit(
            "pass a tweet, --file path, or pipe lines on stdin"
        )

    payloads = [result_payload(text, args.threshold) for text in texts]
    if args.json:
        print(json.dumps(payloads if len(payloads) > 1 else payloads[0], ensure_ascii=False, indent=2))
        return

    for payload in payloads:
        print(payload["text"])
        print(f"  tokens : {payload['tokens']}")
        print(
            f"  label  : {payload['label']}  "
            f"score={payload['score']:.2f}  threshold={args.threshold}"
        )
        if payload["reasons"]:
            for reason in payload["reasons"]:
                print(f"  cue    : {reason}")
        else:
            print("  cue    : (none)")
        print()


if __name__ == "__main__":
    main()
