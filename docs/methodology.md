# Methodology

This page is the experiment protocol as implemented, not as a cleaned-up paper would rewrite it.

## Task

Binary classification: given one tweet-length string, predict sarcastic (`1`) vs not (`0`). No conversation thread, no author graph, no image pixels. The second “modality” is **emoji identity in the same string**, represented with emoji2vec rather than a separate encoder.

## Why emoji2vec at all

Sarcasm on Twitter is often a *tone marker* glued onto otherwise positive words (`love`, `great`, `yay`) plus a face (`😒`, `😑`) or a hashtag (`#not`, `#sarcastictweet`). Word GloVe trained on Twitter already sees some of those faces if they were frequent enough to get a row. emoji2vec is a skip-gram space trained to put Unicode emoji near the words they co-occur with. Averaging it into the same 200-d slot (deep path) or concatenating it as an extra 200-d block (classical path) is a cheap multi-modal fusion: same time step / same tweet, different pretrained table.

## Feature paths

### A. Mean-pooled embeddings (sklearn)

```
ReadOpen
  → for each token, lookup GloVe (200-d) if present
  → mean, or zeros if the tweet has no in-vocab tokens
  → optionally mean of emoji2vec hits, concat → 400-d
  → shared permutation of rows
  → SVC / DT / RF / GBT
```

No n-grams, no TF-IDF, no character features. The lexical shortcuts (`#not`) survive only insofar as those tokens have distinctive vectors.

### B. Sequence + attention (Keras)

```
ReadOpen
  → Keras Tokenizer on train lists
  → pad post to train max length
  → embedding matrix via Preprocess(get_emoji2vec=True|False)
  → PrepModel: frozen Embedding, 2× BiLSTM, Attention, sigmoid
```

Single-modal vs multi-modal is **only** the embedding matrix. The graph is identical. That is a clean ablation: any test gap is “did this token get a nonzero emoji2vec row instead of a zero / GloVe miss.”

## Training details that are in code

| Knob | Value | Where |
| --- | --- | --- |
| Embedding dim | 200 | GloVe Twitter 27B 200-d + emoji2vec |
| Embedding trainable | False | `dl_model.py` |
| LSTM width | 256 each direction | `PrepModel` |
| Dropout | 0.25 then 0.40 / 0.40 | `PrepModel` |
| Loss | binary cross-entropy | `compile` |
| Optimizer | Adam `lr=0.001` | `Adam(lr=lrate)` (Keras 2 kwarg) |
| Output | 1-unit sigmoid | threshold 0.5 at `predict` time in the metrics notebook |
| Sklearn constructors | library defaults | `baseline_models.ipynb` |

What is **not** in the committed Python (only in the notebooks / lost cells):

- epoch count, batch size, early stopping, class weights
- random seeds for TF / NumPy / sklearn
- whether validation was a split of train or the test set reused as `X_val = X_test` (the evaluate notebook does that assignment — do not treat it as a proper val split)

## Evaluation protocol

Three numbers per model family:

1. **Test accuracy / F1** on all 2,000 balanced test tweets.
2. **Subtest accuracy / F1** on the 278 emoji-bearing test tweets.
3. Same metrics for the `_we` (word+emoji) variant.

F1 is sklearn’s default binary F1 on the positive class (sarcastic = 1). That matches a “find the sarcastic tweets” view. On the balanced test set, acc and F1 stay close. On subtest (61.9% positive) F1 runs higher than acc for every model — expected when the positive class is the majority.

Threshold for the BiLSTM is 0.5 on the sigmoid output (`get_metrics_of_models.ipynb`).

## Ablations that exist

| Comparison | What it isolates |
| --- | --- |
| `*_we` vs plain | emoji2vec present vs absent |
| sklearn vs BiLSTM | order + attention vs mean pool |
| test vs subtest | emoji-conditioned slice vs full mix |
| `#not` lexical baseline (`examples/lexical_baseline.py`) | how far surface cues go without embeddings |

There is no “emoji-only” deep model, no late fusion (separate emoji encoder + concat before Dense), and no contextual LM (BERT etc.). Those would be follow-ups, not claims of this archive.

## Preprocessing choices that affect conclusions

1. **Comma join.** `split(',')` then re-join with spaces. Harmless for most tweets; it will smash CSV quoting artifacts and also split decimal-looking commas.
2. **Lowercasing after TweetTokenizer.** Hashtags become `#not` still, emoji stay, `<user>` stays.
3. **Post padding.** Attention bias `b` is length-tied; trailing pads can still receive mass unless the mask is passed. `ReadOpen` does not build a Keras mask. Whether `Embedding` was created with `mask_zero=True` is **no** in `PrepModel`. Pad tokens are real steps. Attention can look at padding. That is a real limitation of the shipped graph.
4. **Shuffling inside `ml_read_data`.** Fine for i.i.d. sklearn fits; do not use it if you need aligned ids across scripts without saving arrays.
5. **Train / test leak.** 48 raw-line collisions, **242 after comma flatten** (the view `ReadOpen` uses). Reported in [dataset.md](dataset.md). A cleaner rerun would drop every flattened test string that appears in train.

## Recommended local rerun (modern TF)

`Adam(lr=...)` is the old kwarg (`learning_rate` now). `tensorflow.python.keras.layers.embeddings.Embedding` is a compatibility import; prefer `tensorflow.keras.layers.Embedding` once. The attention layer’s `add_weight` name interpolation is Keras 2 style. A faithful rerun should pin TF 2.10–2.15 or rewrite the three Python modules against Keras 3.

See [reproduction.md](reproduction.md) for the file-level gaps (missing GloVe, missing SVM/RF pickles, incomplete SavedModel folders).
