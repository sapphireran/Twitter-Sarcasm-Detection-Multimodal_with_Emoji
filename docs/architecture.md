# Architecture

This repository is a 2023 University of Copenhagen Computational Cognitive
Science 2 final project. The question it asks is narrow:

> If you already have word embeddings for a tweet, does adding emoji
> embeddings help a sarcasm classifier?

Two families of models answer that question on the same splits.

## Modalities

| Name | Feature | Shape used by the classifiers |
| --- | --- | --- |
| Single-modal (`W`) | GloVe-Twitter 200d, averaged or sequenced | 200-d bag, or length × 200 sequence |
| Multi-modal (`WE`) | The same GloVe vector plus emoji2vec 200d | 400-d bag, or a sequence whose OOV tokens may be filled from emoji2vec |

`W` means words. `WE` means words + emoji. The deep models do not concatenate
two towers at the classifier. They share one embedding matrix. When
`get_emoji2vec=True`, tokens that miss GloVe are replaced by the mean of any
emoji found inside that token. When `get_emoji2vec=False`, those rows stay
zero. That is the only architectural difference between
`model/best_model_single_modal` and `model/best_model_multi_modal`.

## Classical bag-of-vectors path

```
tweet
  → TweetTokenizer + lowercase          (data_utils.ReadOpen)
  → mean GloVe row over in-vocab tokens (AverageVectorPerTweet)
  → optional mean emoji2vec row         (AverageVectorPerEmoji)
  → concat to 400-d if multi-modal
  → SVM / Decision Tree / Random Forest / Gradient Boosting
```

Missing vocabulary is not fatal. A tweet with no in-vocab tokens becomes a
200-d zero vector. A tweet with no emoji becomes a 200-d zero block in the
concatenated 400-d vector.

The notebooks shuffle the training matrix with `np.random.permutation` inside
`ml_read_data`. Labels stay aligned because the same index order is applied to
`X` and `y`.

## Deep sequence path

```
tweet
  → TweetTokenizer + lowercase
  → Keras Tokenizer → integer ids → post-padding
  → frozen 200-d Embedding (GloVe, optional emoji2vec fill-in)
  → Dropout 0.25
  → Bidirectional LSTM 256, return_sequences=True
  → Dropout 0.4
  → Bidirectional LSTM 256, return_sequences=True
  → Dropout 0.4
  → Raffel attention (attention_layer.Attention)
  → Dense(1, sigmoid)
```

`dl_model.PrepModel` builds that stack. The saved Keras directories under
`model/` are the best checkpoints from the original training runs.

The attention layer is a feed-forward score over time, not scaled dot-product
attention. For a sequence `x` of shape `(batch, steps, features)`:

```
e_t = tanh(x_t · W + b_t)
α   = softmax(e)
c   = Σ_t α_t x_t
```

`W` has shape `(features,)`. `b` has shape `(steps,)`, so the bias is tied to
the padded time index, not to a feature channel. A NumPy walkthrough of those
three lines lives in `sarcasm_lib/attention.py` and
`examples/attention_walkthrough.py`.

## Why the subtest looks easy

`dataset/subtest_*.csv` is not an independent draw from the same distribution.
Every one of its 278 tweets also appears in `dataset/test_*.csv`, and 276 of
them contain emoji that the current extractor sees (the other two contain
misc-symbol glyphs such as ⭕ and regional-indicator flags). It is the
emoji-bearing slice of the official test set.

That is why every model in the original tables jumps on "subtest", and why a
hashtag heuristic can reach 0.89 accuracy there without any neural net. The
cue is on the surface: 134 of 278 subtest tweets carry an explicit sarcasm
hashtag (`#not`, `#sarcastictweet`, `#yeahright`, …), and those markers are
precision 1.0 on both test and subtest.

Read `docs/dataset.md` and `docs/results.md` before treating subtest as a
generalisation number.

## Repository map

| Path | Role |
| --- | --- |
| `data_utils.py` | Original loaders and embedding-matrix builder |
| `dl_model.py` | Bi-LSTM + attention constructor |
| `attention_layer.py` | Keras Raffel attention |
| `baseline_models.ipynb` | Train / reload SVM, DT, RF, GBT |
| `evaluate_loaded_dl_models.ipynb` | Reload the two saved Keras models |
| `get_metrics_of_models.ipynb` | Accuracy, F1, precision, recall, plots |
| `sarcasm_lib/` | Dependency-light helpers for the examples |
| `examples/` | Runnable scripts that do not need GloVe or TensorFlow |
| `docs/` | Dataset, models, results, reproduction |
