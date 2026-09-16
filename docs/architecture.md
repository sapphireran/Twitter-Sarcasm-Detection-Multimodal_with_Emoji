# Architecture

Two pipelines share `data_utils.py` and the same CSV splits. They
differ in how a tweet becomes a vector and in the classifier that
reads that vector.

```
tweet line
    │
    ├─ ReadOpen ── TweetTokenizer, lowercase, commas → spaces
    │
    ├─ classical ── mean GloVe (200-d)
    │                 optional mean emoji2vec (200-d)
    │                 concat → 200-d or 400-d bag
    │                 SVM / DT / RF / GBT
    │
    └─ deep ── Keras Tokenizer + post-padding
                 embedding matrix (GloVe, emoji2vec fallback)
                 Dropout 0.25
                 BiLSTM 256 ── Dropout 0.4
                 BiLSTM 256 ── Dropout 0.4
                 Attention (Raffel 2016)
                 Dense(1, sigmoid)
```

## Classical baselines

Implemented in `baseline_models.ipynb` (training) and
`get_metrics_of_models.ipynb` (accuracy, F1, precision, recall).

Each model is fit twice:

| Suffix in the pickle name | Input | Dim |
| --- | --- | ---: |
| `*_classifier.pkl` | mean GloVe only | 200 |
| `*_classifier_we.pkl` | GloVe mean ∥ emoji2vec mean | 400 |

`we` is “with emoji.” Default sklearn constructors are used (`SVC()`,
`DecisionTreeClassifier()`, `RandomForestClassifier()`,
`GradientBoostingClassifier()`) — no grid search is in the notebook.

**Caveat:** the `except FileNotFoundError` branches in
`baseline_models.ipynb` do not always construct the classifier named
in the section. The RF single-modal fallback builds an `SVC()`, the
GBT single-modal fallback also builds an `SVC()`, and the DT
multi-modal fallback builds an `SVC()`. The **reported** numbers
come from the already-pickled files that were loaded successfully in
2023. Retraining from those fallbacks will not reproduce the tables.

Pickles present in git today: `dt_classifier.pkl`,
`dt_classifier_we.pkl`, `gbt_classifier.pkl`,
`gbt_classifier_we.pkl`. SVM and RF pickles are referenced by the
notebooks but are not in the tree.

## Deep model (`dl_model.PrepModel`)

```
Embedding(vocab, 200, weights=embedding_matrix, trainable=False)
Dropout(0.25)
Bidirectional(LSTM(256, return_sequences=True,
                   kernel_initializer='he_normal',
                   activation='tanh',
                   recurrent_activation='sigmoid'))
Dropout(0.4)
Bidirectional(LSTM(256, return_sequences=True, ... same ...))
Dropout(0.4)
Attention()
Dense(1, activation='sigmoid')
```

- Optimizer: `Adam(lr=0.001)` (Keras 2 keyword `lr`, not
  `learning_rate`).
- Loss: `binary_crossentropy`.
- Metric: `acc`.
- Sequence length in the saved run: 78 (train max after padding).
- Hidden width after each BiLSTM: 512 (256 × 2).
- Trainable parameter count reported by the notebook: 2,510,848.
  The frozen embedding table is **not** in that number.

Single-modal vs multi-modal is **not** a second LSTM tower. Both
SavedModels have the same graph. The difference is how
`Preprocess(..., get_emoji2vec=True|False)` fills rows of
`embedding_matrix` when a token is missing from GloVe but looks like
emoji. See [emoji_modality.md](emoji_modality.md).

## Attention layer

`attention_layer.Attention` follows Raffel et al.,
[Feed-Forward Networks with Attention Can Solve Some Long-Term
Memory Problems](https://arxiv.org/abs/1512.08756).

For an input `x` of shape `(batch, steps, features)`:

1. `e_t = tanh(x_t · W + b_t)` — `W` is a vector of length
   `features`; `b` is per-timestep if `bias=True`.
2. `α = softmax(e)` with a mask applied after `exp`, plus `epsilon`
   in the denominator so an empty mask does not NaN.
3. Output is `Σ_t α_t x_t`, shape `(batch, features)`.

The layer sets `supports_masking = True` but `compute_mask` returns
`None`, so downstream layers do not see the pad mask. Padding is
still ignored inside `call` when Keras supplies a mask.

A NumPy walkthrough that uses the same equations (no TensorFlow):
`examples/attention_walkthrough.py`.

## Why this shape

The coursework bet was:

- Mean-pooled bags are a fair, cheap baseline and make emoji a
  simple extra 200 numbers.
- A bidirectional LSTM can use word order (the `#not` often lands at
  the end; polarity words land earlier).
- Attention lets the network up-weight the tag or the emoji without
  forcing a fixed pooling recipe.

That is also why the subtest exists: if emoji2vec is doing anything,
the gain should show up where emoji are actually on the page.
