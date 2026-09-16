# Architecture

Two families share the same tokenized tweets and the same 200-d tables.

```text
                    ┌─ mean pool words (200) ── sklearn (SVM / DT / RF / GBT)
 tweet tokens ──────┤
                    │
                    ├─ concat(word pool, emoji pool) (400) ── sklearn _we
                    │
                    └─ padded ids + frozen Embedding(200)
                           → Dropout 0.25
                           → BiLSTM 256 (seq)
                           → Dropout 0.4
                           → BiLSTM 256 (seq)
                           → Dropout 0.4
                           → Raffel attention
                           → Dense(1, sigmoid)
```

The sklearn path is order-blind. The recurrent path is not. That gap, more than emoji2vec, is what the 2023 numbers show.

## Attention (`attention_layer.py`)

The layer follows Raffel et al., *Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems* ([arXiv:1512.08756](https://arxiv.org/abs/1512.08756)). It sits on an RNN that returns sequences.

For an input `x` of shape `(batch, steps, features)`:

```text
e_t  = tanh( x_t · W  +  b_t )     # W: (features,), b: (steps,) if bias
a_t  = softmax_t( e )              # mask applied after exp, then ε in the denom
h    = Σ_t a_t x_t                 # (batch, features)
```

Implementation details that matter when you reimplement it:

- `W` is a **vector**, not a matrix. This is a single-head, single-score attention.
- Bias is **per time step**, length `input_shape[1]`. That couples the layer to a fixed `maxlen` (78 in the saved models). Variable-length batches with a different pad length cannot reuse the same `b` weights.
- Masking is supported (`supports_masking = True`) but `compute_mask` returns `None`, so downstream layers do not see a mask.
- Softmax is implemented as `exp` then divide by `sum + epsilon` to avoid NaNs when a row is all masked.

`examples/05_attention_math.py` runs this formula in NumPy on a 3-step toy sequence and checks that (1) weights sum to 1, (2) a mask zeros a step, (3) the context vector is the weighted sum.

## `PrepModel` (`dl_model.py`)

```python
Embedding(count, 200, weights=[embedding_matrix], input_length=l, trainable=False)
Dropout(0.25)
Bidirectional(LSTM(256, he_normal, recurrent_activation='sigmoid',
                   return_sequences=True, activation='tanh'))
Dropout(0.4)
Bidirectional(LSTM(256, ... return_sequences=True ...))
Dropout(0.4)
Attention()
Dense(1, activation='sigmoid')
Adam(lr=0.001), binary_crossentropy, metrics=['acc']
```

`evaluate_loaded_dl_models.ipynb` printed this summary for both saved models (the embedding table is wrapped, so Keras reports 0 params on those wrappers even though 200-d rows exist):

| Layer | Output | Notes |
| --- | --- | --- |
| embedding | `(None, 78, 200)` | frozen at train time in `PrepModel` |
| dropout | `(None, 78, 200)` | 0.25 |
| bidirectional LSTM | `(None, 78, 512)` | 256 × 2 directions |
| dropout | `(None, 78, 512)` | 0.4 |
| bidirectional LSTM | `(None, 78, 512)` | 256 × 2 |
| dropout | `(None, 78, 512)` | 0.4 |
| attention | `(None, 512)` | collapses time |
| dense | `(None, 1)` | sigmoid |

Reported total: **2,510,848** parameters. The notebook’s “trainable = all of them” line is a serialization artifact of the `ModuleWrapper` export; the source `PrepModel` freezes the embedding.

Single-modal and multi-modal saved models have the **same** graph. The difference is which 200-d rows were written for emoji tokens before `fit`.

## Classical baselines

`baseline_models.ipynb` mean-pools with `ml_read_data` and fits:

| File stem | Intended estimator | Feature view |
| --- | --- | --- |
| `svm_classifier` | `SVC()` | 200-d word mean |
| `svm_classifier_we` | `SVC()` | 400-d concat |
| `dt_classifier` | `DecisionTreeClassifier()` | 200-d |
| `dt_classifier_we` | *notebook fallback trains `SVC()`* | 400-d |
| `rf_classifier` | *fallback trains `SVC()`* | 200-d |
| `rf_classifier_we` | `RandomForestClassifier()` | 400-d |
| `gbt_classifier` | *fallback trains `SVC()`* | 200-d |
| `gbt_classifier_we` | `GradientBoostingClassifier()` | 400-d |

The `try: joblib.load` / `except FileNotFoundError` blocks in that notebook have copy-paste bugs: several single-modal fallbacks construct `SVC()` even under a DecisionTree / RF / GBT heading. The **loaded** pickles in `baseline_models/` are whatever was trained in the successful 2023 session (the load path printed “Loaded … from files successfully”). Treat the pickle files + `get_metrics_of_models.ipynb` outputs as the experimental record, not the except-branch constructors.

Only DT and GBT pickles are present in this git tree (`dt_classifier*.pkl`, `gbt_classifier*.pkl`). SVM and RF pickles are referenced by the notebooks but were not uploaded.

## Why 200-d and 400-d

- GloVe-Twitter 27B **200d** is the word table named in the notebooks (`glove.twitter.27B.200d.bin`).
- emoji2vec is also 200-d in `emoji2vec_twitter.bin`.
- Concatenation is the entire “fusion” story for sklearn. There is no gated fusion, cross-attention, or learned modality weight.

A more 2026-typical design would encode the tweet with a frozen transformer and add an emoji-presence / emoji-identity feature. That is out of scope for this archive; the examples stay with mean-pool + a 16-d toy table so they run offline.

## Training hyperparameters that are *not* in `PrepModel`

`dl_model.py` only builds and compiles. Epochs, batch size, and early stopping lived in a notebook that was not uploaded. The saved `model/best_model_*` directories are the only neural artifacts. Do not invent an epoch count in a report; say “as saved in June 2023.”
