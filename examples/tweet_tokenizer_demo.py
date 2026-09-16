#!/usr/bin/env python3
"""Show how the example tokenizer treats sarcasm cues and comma-splitting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.tweet_features import extract_features, normalize_commas, tokenize_tweet


def main() -> None:
    sample_path = Path(__file__).with_name("sample_tweets.json")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))

    print("Tokenizer and feature walkthrough")
    print("=================================")
    print()
    print(
        "data_utils.ReadOpen first does `' '.join(line.strip().split(','))`: "
        "commas become spaces. Hashtags, mentions, and emoji stay intact."
    )
    print()

    for tweet in payload["tweets"]:
        text = tweet["text"]
        features = extract_features(text)
        print(f"[{tweet['id']}] label={tweet['label']}")
        print(f"  raw:        {text}")
        print(f"  commas:     {normalize_commas(text)}")
        print(f"  tokens:     {tokenize_tweet(text)}")
        print(
            "  cues:       "
            f"#not={features.not_hashtag} sarcasm_tag={features.sarcasm_hashtag} "
            f"love={features.love_word} contrast={features.contrast_cue} "
            f"emoji={features.emoji_count} elongated={features.elongated_count}"
        )
        print(f"  notes:      {tweet['notes']}")
        print()


if __name__ == "__main__":
    main()
