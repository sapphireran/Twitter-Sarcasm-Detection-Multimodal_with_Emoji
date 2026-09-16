# Method

Two representation families share the same labels and the same three splits.

1. **Bag of averaged vectors** for sklearn baselines
2. **Frozen embedding + stacked BiLSTM + attention** for the deep model

Both families have a text-only (`W`) variant and a text+emoji (`WE`) variant.

## Tokenization

`data_utils.ReadOpen`:

1. Read a line as UTF-8 (`errors="replace"`).
2. Replace commas with spaces (`' '.join(line.strip().split(','))`).
3. Tokenize with NLTK `TweetTokenizer`.
4. Lowercase every token.

That tokenizer keeps hashtags, emoji, and emoticons as tokens instead of stripping them. The Keras path then feeds the token lists to `keras_preprocessing.text.Tokenizer`, which builds an integer vocabulary and maps out-of-vocabulary test tokens to `0` (the pad / OOV id).

`examples/sarcasm_lib/tokenize.py` is a smaller reimplementation used by the docs examples. It is not byte-identical to NLTK, but it keeps the same token classes: words, `#hashtags`, `<user>`, `<url>`, and emoji.

## Text-only vs multimodal embeddings

### Classical models (`ml_read_data`)

For each tweet:

- `AverageVectorPerTweet` averages every token that hits the GloVe vocabulary. Missing tweets become a 200-d zero vector.
- `AverageVectorPerEmoji` averages every token that hits the emoji2vec vocabulary. Missing tweets become a 200-d zero vector.

`W` features are the 200-d GloVe average. `WE` features are the 400-d concatenation `[glove_avg ; emoji_avg]`.

Both matrices are shuffled with `numpy.random.permutation` **without a seed**. Reloading the notebook retrains on a different row order. Saved pickles in `baseline_models/` freeze one such run.

### Sequence model (`Preprocess`)

`Preprocess` builds a `(n_rows, 200)` embedding matrix and a post-padded index matrix.

For each tokenizer word `w`:

1. If `w` is in GloVe, copy that row.
2. Else run `emoji.emoji_list(w)` / `emoji.is_emoji` and, if any emoji characters remain, average their emoji2vec rows (`get_emoji2vec=True`).
3. Else write a 200-d zero row.

The single-modal checkpoint is the same architecture with `get_emoji2vec=False` (emoji tokens that miss GloVe stay zero). The multimodal checkpoint fills those rows from `emoji2vec_twitter.bin`.

**Quirk:** the embedding matrix is allocated as `zeros((count, 200))` where `count` is the *number of training tweets*, not `len(tokenizer.word_index) + 1`. This only works if the vocabulary is smaller than the tweet count (true for 39,780 training lines). A correct allocation is `vocab_size = len(word_index) + 1`.

## Deep architecture

`dl_model.PrepModel` builds:

| Layer | Output width | Notes |
| --- | ---: | --- |
| `Embedding(count, 200, trainable=False)` | 200 | Frozen GloVe / emoji2vec rows |
| `Dropout(0.25)` | 200 | |
| `Bidirectional(LSTM(256, return_sequences=True))` | 512 | `he_normal`, `tanh` / `sigmoid` |
| `Dropout(0.4)` | 512 | |
| `Bidirectional(LSTM(256, return_sequences=True))` | 512 | Same LSTM settings |
| `Dropout(0.4)` | 512 | |
| `Attention()` | 512 | Sequence → tweet vector |
| `Dense(1, sigmoid)` | 1 | P(sarcastic) |

Compiled with `Adam(lr=0.001)`, `binary_crossentropy`, and `acc`. The saved models in `model/best_model_*` report about 2.51M trainable parameters (the embedding table is stored but marked non-trainable in the construction script; the serialized summaries show the embedding wrapped as `ModuleWrapper` with 0 listed params).

Maximum padded length on the recorded run was **78** time steps.

## Attention layer

`attention_layer.Attention` is a Keras 2 port of feed-forward attention over time ([Raffel and Ellis, 2015](https://arxiv.org/abs/1512.08756)).

For input `x` of shape `(batch, steps, features)`:

```text
e_t = tanh(x_t · W + b_t)          # W is (features,), b is (steps,) if bias=True
a_t = exp(e_t)
a_t = a_t * mask_t                 # optional
a   = a / (sum_t a_t + ε)
h   = sum_t a_t * x_t              # (batch, features)
```

`W` is shared across time. `b` is a per-timestep bias the width of the *padded length*, which is why `build()` uses `input_shape[1]`. That ties the layer to a fixed pad length after the first `build`.

`examples/sarcasm_lib/attention.py` and `examples/attention_demo.py` replay the same equations in NumPy, including the mask path and the `ε` guard against an all-zero softmax.

## Classical baselines

`baseline_models.ipynb` fits, for each of SVM / decision tree / random forest / gradient boosting:

- a `W` model on 200-d averages
- a `WE` model on 400-d concatenations

Default sklearn hyperparameters (the constructors are called with no arguments). Models are `joblib.dump`-ed under `baseline_models/`.

**Quirk in the “train if pickle missing” branches:** several fallbacks construct the wrong estimator (`DecisionTree` multimodal → `SVC()`, `RandomForest` text-only → `SVC()`, `GradientBoosting` text-only → `SVC()`). The printed numbers in the notebook come from *loaded* pickles, so they are not produced by those fallbacks. If you delete the pickle directory and re-run the notebook as written, you will not recreate the published table.

Only decision-tree and gradient-boosting pickles are in this checkout. SVM and random-forest pickles are referenced by the notebooks but were not uploaded.

## Why a multimodal split exists

Emoji are sparse in train/test (~14%) and almost complete in subtest (~100%). A model can ignore emoji entirely and still look strong on the 2,000-row test set. The subtest is the place where a 200-d emoji average, or an emoji2vec row inside the embedding table, is allowed to matter.

The recorded pattern matches that design:

- SVM **drops** 0.6 points when emoji averages are concatenated on the full test set (0.769 → 0.763), then **gains** on subtest (0.813 → 0.824)
- BiLSTM + attention **gains** on both (0.864 → 0.874 test, 0.867 → 0.892 subtest)

Averaging emoji into a second 200-d bag is a weak fusion method: most tweets contribute a zero block, so the extra 200 dimensions are empty. The sequence model can keep emoji on their own timesteps and let attention up-weight them.

## Evaluation protocol

Metrics in the notebooks:

- Accuracy
- Binary F1, precision, and recall with sklearn defaults (`pos_label=1`)

The deep-model `evaluate()` path reports Keras `acc` on the same padded matrices. The F1 numbers come from `model.predict` rounded / thresholded in `get_metrics_of_models.ipynb`.

There is no k-fold on train. Test is used both as a validation printout (`X_val = X_test` in `evaluate_loaded_dl_models.ipynb`) and as the reported test set. Treat the published scores as a single train/test snapshot, not as a nested-CV estimate.

## Lightweight docs baseline

`examples/lexical_baseline.py` trains a multinomial Naive Bayes classifier from scratch on:

- word tokens
- hashtag tokens
- emoji tokens
- a small set of boolean cues (`has_not_hashtag`, `has_explicit_sarcasm_tag`, `has_emoji`, `has_user`)

It exists so the docs tree has a number you can regenerate in this environment without GloVe. It is intentionally leaky: `#not` is a gold-adjacent feature. That is the point of the writeup — a large fraction of the original label is hashtag-visible.
