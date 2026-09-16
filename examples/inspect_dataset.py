#!/usr/bin/env python3
"""Print split sizes, cue rates, and a few labeled rows.

Usage (from the repo root):

    python3 examples/inspect_dataset.py
    python3 examples/inspect_dataset.py --dataset-dir dataset --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.dataset_io import (  # noqa: E402
    cue_stats,
    load_split,
    split_sizes,
    top_hashtags,
)
from examples.lib.lexical_features import hashtag_rule_predict  # noqa: E402


def _example_rows(split, want_label: int, need_emoji: bool, limit: int) -> list[str]:
    from examples.lib.dataset_io import EMOJI_RE

    rows: list[str] = []
    for sentence, label in split.labeled():
        if label != want_label:
            continue
        if need_emoji and not EMOJI_RE.search(sentence):
            continue
        rows.append(sentence)
        if len(rows) >= limit:
            break
    return rows


def _hashtag_rule_stats(split) -> dict[str, float | int]:
    """How much of the gold label is just `#not` / `#sarcasm*` / `#yeahright`."""
    tp = fp = fn = tn = 0
    for sentence, label in split.labeled():
        pred = hashtag_rule_predict(sentence)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1
    n = max(tp + fp + fn + tn, 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / n,
        "precision": prec,
        "recall": rec,
        "sarcastic_with_tag": tp,
        "sarcastic_without_tag": fn,
    }


def report(dataset_dir: Path) -> dict:
    payload: dict = {"dataset_dir": str(dataset_dir), "splits": {}}
    for name in ("train", "test", "subtest"):
        split = load_split(name, dataset_dir)
        sizes = split_sizes(split)
        cues = cue_stats(split)
        payload["splits"][name] = {
            "sizes": sizes,
            "cues": cues,
            "top_hashtags": top_hashtags(split, n=8),
            "hashtag_rule": _hashtag_rule_stats(split),
            "sample_sarcastic": _example_rows(split, 1, False, 2),
            "sample_literal": _example_rows(split, 0, False, 2),
        }
    return payload


def render(payload: dict) -> str:
    lines = [f"dataset dir: {payload['dataset_dir']}", ""]
    for name, block in payload["splits"].items():
        s = block["sizes"]
        c = block["cues"]
        lines.append(f"== {name} ==")
        lines.append(
            f"rows {s['rows']:,}   literal {s['literal']:,}   "
            f"sarcastic {s['sarcastic']:,}"
        )
        lines.append(
            f"tokens mean {c['mean_tokens_x100'] / 100:.2f}  "
            f"min {c['min_tokens']}  max {c['max_tokens']}"
        )
        lines.append(
            f"chars min {c['min_chars']}  max {c['max_chars']}"
        )
        lines.append(
            f"with # {c['with_hashtag']:,}   with emoji {c['with_emoji']:,}   "
            f"with user {c['with_user']:,}   sarc-ish tag {c['with_sarc_tag']:,}"
        )
        if c["top_hashtag"]:
            lines.append(
                f"top hashtag {c['top_hashtag']} × {c['top_hashtag_count']}"
            )
        rule = block["hashtag_rule"]
        lines.append(
            f"hashtag-rule acc {rule['accuracy']:.3f}  "
            f"prec {rule['precision']:.3f}  rec {rule['recall']:.3f}  "
            f"(tagged sarcastic {rule['sarcastic_with_tag']:,} / "
            f"missed {rule['sarcastic_without_tag']:,} / "
            f"false tags {rule['fp']:,})"
        )
        tags = ", ".join(f"{t}={n}" for t, n in block["top_hashtags"])
        if tags:
            lines.append(f"hashtags: {tags}")
        if block["sample_sarcastic"]:
            lines.append("sarcastic e.g.: " + block["sample_sarcastic"][0][:140])
        if block["sample_literal"]:
            lines.append("literal   e.g.: " + block["sample_literal"][0][:140])
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "dataset",
        help="directory with *_sentence.csv and *_label.csv",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of text",
    )
    args = parser.parse_args(argv)
    payload = report(args.dataset_dir)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
