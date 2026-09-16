# Architecture

Two model families share the same tokenized tweets and the same two embedding tables. They differ in how they fuse word and emoji information.

```text
                    ┌─ mean GloVe (200) ──────────────┐
 tokens ─┬─ GloVe ──┤                                 ├─ 400-d → SVM / DT / RF / GBT
         │          └─ mean emoji2vec (200) ──────────┘
         │
         └─ sequence of 200-d rows (GloVe or emoji2vec per token)
                → Dropout 0.25
                → Bi-LSTM 256 (sequences)
                → Dropout 0.4
                → Bi-LSTM 256 (sequences)
                → Dropout 0.4
                → Attention (Raffel-style)
                → Dense(1, sigmoid)
```

## Classical baselines

Trained in `baseline_models.ipynb`, scored again in `get_metrics_of_models.ipynb`. Each algorithm is fit twice:

| Suffix in the pickle names | Features | Dim |
| --- | --- | ---: |
| `*_classifier.pkl` (single-modal) | mean GloVe only | 200 |
| `*_classifier_we.pkl` (multi-modal, “with emoji”) | `concat(mean GloVe, mean emoji2vec)` | 400 |

Algorithms and what is actually on disk in `baseline_models/`:

| Algorithm | Single-modal pickle | Multi-modal pickle | Notes |
| --- | --- | --- | --- |
| SVM (`sklearn.svm.SVC`) | not in this snapshot | not in this snapshot | Notebooks also refer to `svm_model.pkl` / `svm_classifier.pkl`. |
| Decision tree | `dt_classifier.pkl` | `dt_classifier_we.pkl` | Default `DecisionTreeClassifier`. |
| Random forest | not in this snapshot | not in this snapshot | Notebooks load `rf_classifier*.pkl`. |
| Gradient boosting | `gbt_classifier.pkl` | `gbt_classifier_we.pkl` | Default `GradientBoostingClassifier`. |

The notebooks try to load pickles and only refit on `FileNotFoundError`. The 2023 numbers in [evaluation.md](evaluation.md) come from those saved models, not from a fresh unseeded fit.

Default sklearn hyperparameters were used (the cells call `SVC()`, `DecisionTreeClassifier()`, `RandomForestClassifier()`, `GradientBoostingClassifier()` with no grid search). These are *reference* baselines: strong enough to show that a frozen Bi-LSTM is doing more than mean-pooling, not an exhaustive AutoML sweep.

## Bi-LSTM + Attention

`dl_model.PrepModel(count, embedding_matrix, l, lrate=0.001)` builds one Sequential network:

| Layer | Configuration | Role |
| --- | --- | --- |
| `Embedding` | `count × 200`, weights = `embedding_matrix`, `trainable=False`, `input_length=l` | Frozen GloVe / emoji2vec lookup. |
| `Dropout` | 0.25 | Noise on the embedding sequence. |
| `Bidirectional(LSTM 256)` | `he_normal`, `tanh` / `sigmoid` recurrent, `return_sequences=True` | Forward + backward context. Output 512-d per step. |
| `Dropout` | 0.4 | |
| `Bidirectional(LSTM 256)` | same as above | Deeper temporal features. Still 512-d per step. |
| `Dropout` | 0.4 | |
| `Attention` | Raffel et al. 2015, implemented in `attention_layer.py` | Weighted sum over time → one 512-d tweet vector. |
| `Dense(1, sigmoid)` | | P(sarcastic). |
| Optimizer | `Adam(lr=0.001)` | |
| Loss | `binary_crossentropy` | |

Saved runs:

| Directory | Condition | Test acc (notebook) | Subtest acc (notebook) |
| --- | --- | ---: | ---: |
| `model/best_model_single_modal` | GloVe only; emoji / OOV rows are zeros | 0.8635 | 0.8669 |
| `model/best_model_multi_modal` | GloVe + emoji2vec in the same 200-d matrix | 0.8735 | 0.8921 |

The evaluation notebook prints `sequential_5` (single-modal) and `sequential_6` (multi-modal). Both summaries show a padded length of **78** time steps.

### Why attention instead of a final LSTM state

Sarcasm in this data is often a *local* clash: a positive predicate (`love`, `great`, `can't wait`) plus a downshift (`#not`, `😒`, `dirty`, `3am`). A single final hidden state can wash that clash out, especially after 78 steps of padding. Attention lets the network put weight on the clash tokens. `examples/attention_demo.py` runs the exact scoring formulas from `attention_layer.py` on a toy sequence so you can see a downshift token receive more mass.

### The attention layer, precisely

`Attention` follows Raffel, Luong, Liu, Weiss, and Eck, *Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems* (arXiv:1512.08756), and the common Keras 2 port:

1. Score each time step `e_t = tanh(x_t · W + b_t)`.
2. `α = softmax(e)` with a mask and `epsilon` in the denominator to avoid NaNs.
3. Return `Σ_t α_t x_t`.

Implementation details that affect reuse:

- `W` has shape `(features,)`, not `(features, 1)`. The code `K.dot(x, K.expand_dims(self.W))` is a batched dot against that vector.
- `b` has shape `(timesteps,)`, **not** `(features,)`. The layer is bound to the padded length used at build time.
- `compute_mask` returns `None`, so later layers do not see the original padding mask. Padding steps can still receive a little attention mass; the softmax epsilon trick is what keeps training numerically stable.
- `supports_masking = True` so an upstream `Embedding` mask can zero the *unnormalized* `exp(e)` terms.

`examples/attention_demo.py` reimplements those three equations in NumPy and checks them against a few hand-built sequences. It does not load the saved Keras models.

## Single-modal vs multi-modal, restated

| | Single-modal | Multi-modal |
| --- | --- | --- |
| sklearn | 200-d mean GloVe | 400-d concat |
| Bi-LSTM | 200-d sequence; OOV / emoji → `0` | 200-d sequence; OOV / emoji → mean emoji2vec |
| What can represent `love` then `😒` | Only if `😒` somehow sat in GloVe (it does not) | Yes: two neighbouring non-zero rows |
| Trainable params | LSTM + Attention + Dense | Same. Only the *frozen matrix* changes. |

The multi-modal Bi-LSTM therefore has **the same number of trained weights** as the single-modal one. The 1.0-point test-set gain and the 2.5-point subtest gain (see [evaluation.md](evaluation.md)) are feature-quality gains, not extra capacity.

## What is *not* in the architecture

- No character CNN, no BERT, no hashtag-special embedding.
- No explicit “emoji channel” after the embedding layer. Fusion happens *inside* the 200-d / 400-d features.
- No class weights, label smoothing, or focal loss.
- No scheduled sampling or CRF. This is straight binary classification.

Those omissions are appropriate for a 2023 CCS2 final project. If you compare a transformer number to the table in [evaluation.md](evaluation.md), say so: you are no longer comparing against this architecture.
