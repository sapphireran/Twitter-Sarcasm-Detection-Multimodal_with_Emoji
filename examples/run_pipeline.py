#!/usr/bin/env python3
"""Walk one sarcastic tweet through tokenize → average → concatenate.

Also prints the illustrative pair ``i love mondays`` vs ``i love mondays 😒 #not``
so the emoji half is visible as the only changing channel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.average_vectors import average_channel, multimodal_features
from examples.fusion import channel_norms, cosine
from examples.hash_embeddings import default_tables
from examples.tokenize import tweet_tokenize
from examples.toy_corpus import TOY_LABELS, TOY_TWEETS, illustrative_pair, labeled_rows


def _short(vec: np.ndarray, n: int = 6) -> str:
    head = ", ".join(f"{x:+.3f}" for x in vec[:n])
    return f"[{head}, …] dim={vec.size}"


def main() -> int:
    text_table, emoji_table = default_tables(dim=32)

    print("=== toy corpus ===")
    print(f"{len(TOY_TWEETS)} hand-written tweets "
          f"({sum(TOY_LABELS)} sarcastic / {len(TOY_LABELS) - sum(TOY_LABELS)} literal)")
    print()

    demo = "i just love getting shots 💉 #sarcastictweet"
    tokens = tweet_tokenize(demo)
    print("=== stage 1: tokenize (ReadOpen rules) ===")
    print(f"raw:    {demo}")
    print(f"tokens: {tokens}")

    text_vec = average_channel(tokens, text_table)
    emoji_vec = average_channel(tokens, emoji_table)
    print()
    print("=== stage 2: mean-pool each channel (AverageVectorPer*) ===")
    print(f"text  in-vocab: {[t for t in tokens if text_table.known(t)]}")
    print(f"emoji in-vocab: {[t for t in tokens if emoji_table.known(t)]}")
    print(f"x_text  {_short(text_vec)}")
    print(f"x_emoji {_short(emoji_vec)}")

    tokenized = [tweet_tokenize(t) for t in TOY_TWEETS]
    x_text, x_multi = multimodal_features(tokenized, text_table, emoji_table)
    print()
    print("=== stage 3: early fusion (ml_read_data concatenate) ===")
    print(f"X_text  shape {x_text.shape}   X_multi shape {x_multi.shape}")

    sarcastic, literal = illustrative_pair()
    i_s = TOY_TWEETS.index(sarcastic)
    i_l = TOY_TWEETS.index(literal)
    print()
    print("=== illustrative pair: only the emoji half should move ===")
    print(f"sarcastic: {sarcastic!r}")
    print(f"literal:   {literal!r}")
    print(f"cosine(text, text)   {cosine(x_text[i_s], x_text[i_l]):.3f}")
    print(f"cosine(multi, multi) {cosine(x_multi[i_s], x_multi[i_l]):.3f}")
    s_text_n, s_emoji_n = channel_norms(x_multi[i_s])
    l_text_n, l_emoji_n = channel_norms(x_multi[i_l])
    print(f"sarcastic channel L2  text={s_text_n:.3f}  emoji={s_emoji_n:.3f}")
    print(f"literal   channel L2  text={l_text_n:.3f}  emoji={l_emoji_n:.3f}")
    print(
        "literal emoji L2 is 0 because AverageVectorPerEmoji returns zeros "
        "when no emoji is in-vocab — same as data_utils.py."
    )

    print()
    print("=== labeled rows ===")
    for text, label in labeled_rows():
        mark = "SARC" if label == 1 else "lit "
        print(f"  [{mark}] {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
