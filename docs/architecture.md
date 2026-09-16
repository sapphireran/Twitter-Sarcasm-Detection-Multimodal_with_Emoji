# Model architecture

Two classifier families share the same tokenized tweets and then diverge.

## 1. Mean-pooled baselines

After `ml_read_data`, each tweet is one vector.

```
tokens ──► mean GloVe (200-d) ──► SVM / DecisionTree / RandomForest / GBT
                 │
                 └── concat mean emoji2vec (200-d) ──► same four models
```

The notebooks instantiate the sklearn defaults (`SVC()`,
`DecisionTreeClassifier()`, `RandomForestClassifier()`,
`GradientBoostingClassifier()`). There is no grid search in the checked-in
history. The pickled files under `baseline_models/` are those default fits
from June 2023.

A few `except FileNotFoundError` fallbacks in `baseline_models.ipynb`
accidentally construct an `SVC()` while claiming to retrain a tree or a
boosting model. Those branches only run when the pickle is missing; the
checked-in pickles were produced by the intended estimators.

## 2. Bi-LSTM + attention

`dl_model.PrepModel(count, embedding_matrix, l, lrate=0.001)` builds:

```
Embedding(count, 200, weights=embedding_matrix, trainable=False, input_length=l)
  → Dropout(0.25)
  → Bidirectional(LSTM(256, return_sequences=True,
                       kernel_initializer='he_normal',
                       activation='tanh',
                       recurrent_activation='sigmoid'))
  → Dropout(0.4)
  → Bidirectional(LSTM(256, return_sequences=True, … same …))
  → Dropout(0.4)
  → Attention()
  → Dense(1, activation='sigmoid')
```

Compiled with `Adam(lr=0.001)` and `binary_crossentropy`.

Shapes, assuming pad width `L` (78 on the 2023 training run):

| Layer | Output |
| --- | --- |
| Embedding | `(B, L, 200)` |
| Bi-LSTM 1 | `(B, L, 512)` |
| Bi-LSTM 2 | `(B, L, 512)` |
| Attention | `(B, 512)` |
| Dense | `(B, 1)` |

Both LSTMs return sequences so attention can see every timestep. The
recurrent activation is the hard-sigmoid / sigmoid used by the Keras CuDNN
compatible path (`tanh` output, `sigmoid` forget/input/output gates).

The embedding table is frozen. All task learning happens in the recurrent
stack, the attention weights, and the final logistic unit. That keeps the
GloVe / emoji2vec geometry intact and makes the single- vs multi-modal
comparison a comparison of **which rows were filled**, not of a fine-tuned
embedding space.

## Attention layer

`attention_layer.Attention` follows Raffel et al.,
[Feed-Forward Networks with Attention Can Solve Some Long-Term Memory
Problems](https://arxiv.org/abs/1512.08756). It is a timestep scoring
layer, not multi-head self-attention.

For an input `x` of shape `(B, T, H)`:

```
e_t     = tanh( x_t · W  +  b_t )     # W ∈ R^H, b ∈ R^T
α_t     = softmax_t(e)                # mask applied after exp, ε in the denominator
context = Σ_t α_t ⊙ x_t               # (B, H)
```

Implementation details that matter if you port the layer:

1. **`W` is a vector, not a matrix.** The score is a dot product with the
   hidden state, then a tanh. There is no learned query vector beyond `W`.
2. **`b` is length `T`, not length `H`.** The bias is per timestep of the
   *training* pad width. That is why `build()` uses `input_shape[1]`. A
   sequence that is padded to a different `T` at load time will not match
   the stored bias. The 2023 models were trained and evaluated at the same
   `L`.
3. **Masking.** `supports_masking = True`, but `compute_mask` returns
   `None`, so downstream layers do not see a mask. The mask is only used
   inside `call` to zero the exponential scores of padded steps before
   renormalizing.
4. **Numerical guard.** The softmax denominator is
   `sum(α) + K.epsilon()` so a fully masked row does not become `NaN`.

`examples/attention_numpy.py` reimplements this formula with NumPy so you
can print `α` for a toy sequence without TensorFlow.

## Why attention instead of the last hidden state

Sarcasm cues are sparse. A trailing `#not` or a single emoji can flip the
label while the earlier tokens look sincere. A last-timestep readout from
an LSTM puts most of the burden on the recurrent state to carry that cue
to the end. Attention lets the classifier put weight directly on the
informative steps.

The 2023 notebooks do not export per-tweet attention maps. The NumPy
walkthrough is the place to see the mechanism; restoring maps from the
saved Keras models would require the missing `variables/` shards.

## Hyperparameters that were not swept

The checked-in `PrepModel` uses one setting:

| Knob | Value |
| --- | --- |
| Embedding dim | 200 (GloVe Twitter / emoji2vec) |
| LSTM width (per direction) | 256 |
| Stack depth | 2 Bidirectional LSTMs |
| Input dropout | 0.25 |
| Recurrent-stack dropout | 0.4 after each Bi-LSTM |
| Optimizer | Adam, learning rate `1e-3` |
| Loss | binary cross-entropy |
| Output | 1-unit sigmoid |

If you retrain, treat those as the course-project baseline rather than an
optimised configuration.

## Known implementation notes

- `PrepModel` imports `Embedding` twice (Keras and `tensorflow.python.keras`).
  The layer that is actually added is the first import. This is leftover
  from a TF 1.x / 2.x transition and is harmless if both resolve.
- `Adam(lr=...)` is the old argument name. Current Keras wants
  `learning_rate`.
- `Embedding(..., count)` uses tweet count as the vocabulary size. Index
  `0` is padding; Keras `Tokenizer` assigns words from `1`. As long as
  `max(word_index) < count` the table is large enough, with unused rows
  at the end.
- Saved models under `model/best_model_*` contain `saved_model.pb` and
  `keras_metadata.pb` only. Weight shards were not uploaded, so
  `tf.keras.models.load_model` cannot restore them from this clone.
