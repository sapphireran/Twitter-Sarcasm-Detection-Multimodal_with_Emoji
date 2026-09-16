# Emoji modality

“Multi-modal” in this project does **not** mean image pixels or an
audio stream. It means a second pretrained embedding table whose keys
are emoji characters, trained (in the original emoji2vec work) so that
emoji sit near the words they co-occur with.

Two binary tables are in the repo:

| File | Size | Used as |
| --- | ---: | --- |
| `emoji2vec_twitter.bin` | 1.3 MB | primary table in the notebooks |
| `emoji2vec.bin` | 2.0 MB | generic emoji2vec dump, not wired into the notebooks |

Word vectors (`glove.twitter.27B.200d.bin`) are **not** in git.

## Classical path: two means, then concat

`AverageVectorPerTweet` walks tokens and averages every row found in
the GloVe `KeyedVectors`. Tokens that miss the table are skipped. If
nothing hits, the tweet becomes `zeros(200)`.

`AverageVectorPerEmoji` does the same lookup against emoji2vec. A
tweet with no emoji (or with emoji missing from the table) is again
`zeros(200)`.

`ml_read_data` then concatenates:

```
[ g1 ... g200 | e1 ... e200 ]   → 400-d
```

The sklearn models therefore see emoji as an extra block of features,
not as tokens inside the sentence. Order between words and emoji is
discarded twice: once by mean-pooling words, once by mean-pooling
emoji.

A toy version of this exact concat, with a 8-d fake GloVe and a 8-d
fake emoji table, is `examples/toy_embedding_fusion.py`.

## Deep path: one matrix, two sources

`Preprocess` builds a single `(vocab, 200)` matrix for the Embedding
layer.

For each tokenizer index `i` and token `word`:

1. If `word` is in GloVe, copy that row.
2. Else run `emoji.emoji_list(word)`, keep characters that
   `emoji.is_emoji` accepts, look each up in emoji2vec, and average.
3. If `get_emoji2vec` is `False`, or the lookup throws, write zeros.
4. Count misses in `nf` (printed by some notebook cells as
   “words not found”).

So the LSTM still reads a sequence of 200-d vectors. Emoji are not a
parallel tower; they occupy the same slots as words. The
single-modal run is the same graph with those fallback rows zeroed.

That is why `model/best_model_single_modal` and
`model/best_model_multi_modal` have identical `summary()` text in
`evaluate_loaded_dl_models.ipynb`. The architecture string cannot
tell you which embedding table was used.

## What emoji actually do on these splits

- Train: 5,470 / 39,780 tweets match a common-emoji Unicode regex
  (~13.8%).
- Test: 277 / 2,000 (~13.9%).
- Subtest: 277 / 278 (~99.6%) — constructed to be the emoji slice.

On the full test set the multi-modal lift is small (Bi-LSTM +0.010
accuracy, random forest +0.0035). On the subtest the same Bi-LSTM
lift is +0.025 accuracy and +0.017 F1. That pattern matches the
construction: emoji2vec can only move the score when emoji rows are
nonzero.

## Worked toy numbers

Suppose a tweet tokenizes to `['i', 'love', 'this', '😒', '#not']`.

Classical single-modal:

```
mean(GloVe[i], GloVe[love], GloVe[this], GloVe[#not])     # 😒 dropped if not in GloVe
```

Classical multi-modal:

```
mean(those GloVe rows)  ∥  mean(emoji2vec[😒])
```

Deep multi-modal embedding rows:

```
i, love, this, #not  ← GloVe
😒                   ← emoji2vec
```

Deep single-modal:

```
i, love, this, #not  ← GloVe
😒                   ← zeros(200)
```

The attention walkthrough (`examples/attention_walkthrough.py`) uses
a short sequence like this and shows the softmax putting mass on the
late `#not` / emoji steps.
