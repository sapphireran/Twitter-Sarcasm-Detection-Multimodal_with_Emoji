# 03 — Architecture

## Tokenization and embedding lookup

`data_utils.Preprocess` fits a Keras `Tokenizer` on the already-split train
tokens, converts each tweet to an integer sequence, and post-pads. The
embedding matrix has one row per tokenizer index (plus row 0 for padding)
and 200 columns.

Fill order for row `i` of token `w`:

1. If `w` is in GloVe-Twitter, copy that 200-d vector.
2. Else run `emoji.emoji_list(w)` and keep characters for which
   `emoji.is_emoji` is true. Average their emoji2vec rows if
   `get_emoji2vec=True`.
3. Else write zeros. A counter `nf` records how often the emoji branch
   throws (missing key in emoji2vec).

The sklearn path is coarser. `AverageVectorPerTweet` / `AverageVectorPerEmoji`
walk the token list, skip OOV tokens, and average. An all-OOV tweet becomes
a 200-d zero. `ml_read_data` then concatenates the two averages into 400-d
for the WE models. Both views are shuffled with the same `numpy.random.permutation`
index so W and WE rows stay aligned.

Gensim 3.x API: the original code uses `if token in model.vocab` and
`model[token]`. Gensim 4 renamed `vocab` to `key_to_index`. That is the
first thing that breaks if you replay the notebooks on a current install
(see [05-reproduction.md](05-reproduction.md)).

## Neural net (`dl_model.PrepModel`)

```
Embedding(vocab, 200, weights=E, trainable=False, input_length=L)
Dropout(0.25)
Bidirectional(LSTM(256, he_normal, tanh, rec_act=sigmoid, return_sequences=True))
Dropout(0.4)
Bidirectional(LSTM(256, he_normal, recurrent_activation='sigmoid', return_sequences=True))
Dropout(0.4)
Attention()
Dense(1, sigmoid)
Adam(lr=0.001), binary_crossentropy, metric=acc
```

Hidden width after each BiLSTM is 512. The 2023 `model.summary()` reports
**2,510,848 trainable parameters**. The frozen embedding is wrapped in a
`ModuleWrapper` in the SavedModel, which is why the summary prints those
layers as having 0 parameters — the GloVe matrix is loaded as a weight but
was marked non-trainable, and the export path used TF’s compatibility
wrappers.

`evaluate_loaded_dl_models.ipynb` logs sequence length 78 at inference,
matching the padded train width.

### Attention layer

`attention_layer.Attention` follows Raffel et al. 2016:

```
e_t = tanh(h_t · W + b)          # W in R^{512}, b in R^{T}
a   = softmax(e)                 # ε added to the sum
c   = Σ_t a_t h_t                # in R^{512}
```

Masking is supported: after `exp`, masked steps are multiplied by 0 and the
distribution is renormalized. `compute_mask` returns `None`, so downstream
layers do not see a mask. There is **no `get_config`**, which is why
`tf.keras.models.load_model` in TF 2.11+ may refuse the custom object
unless you pass `custom_objects` and/or `compile=False`.

A NumPy twin lives in `examples/common/attention.py` and is exercised by
`examples/05_toy_attention.py`.

## Sklearn baselines

Notebook `baseline_models.ipynb` trains four families, each in a W and a
WE variant, default sklearn hyperparameters:

| Family | Estimator in the successful load path |
| --- | --- |
| SVM | `SVC()` (RBF, C=1) |
| Decision tree | `DecisionTreeClassifier()` |
| Random forest | `RandomForestClassifier()` |
| Gradient boosting | `GradientBoostingClassifier()` |

The `except FileNotFoundError` branches in that notebook are copy-paste
noisy (the RF single-modal fallback constructs an `SVC()`, the DT
multimodal fallback also constructs an `SVC()`). The **loaded** pickles
are what the reported numbers correspond to. Only DT and GBT pickles are
present in `baseline_models/` in this git snapshot; SVM and RF pickles were
referenced by the notebooks but not uploaded.

Feature dimension:

* W: 200 (mean GloVe)
* WE: 400 (mean GloVe ‖ mean emoji2vec)

## Training protocol (from filenames and notebooks)

The best neural checkpoints were saved under names that already contain the
metrics:

```
best_model_w_0.8634999990463257_sub_0.866906464099884
best_model_we_0.8734999895095825_sub_0.8920863270759583
```

Those directories are not in git; `model/best_model_single_modal` and
`model/best_model_multi_modal` are the renamed copies the evaluation
notebook loads. This snapshot only has `keras_metadata.pb` inside each
folder (the weight shards were not uploaded), so you cannot evaluate the
SavedModels from git alone.

Optimizer note: `Adam(lr=0.001)` uses the Keras 2 argument name. Keras 3
wants `learning_rate`.

## Why attention instead of a last hidden state

Sarcastic tweets in this collection are short, but the cue is often a
*single* token: `#not`, a face, or a polarity flip. Mean-pooling a BiLSTM
dilutes that token; a last-state readout depends on the pad direction
(here: post-padding, so the last real token is *not* at the final step).
Attention with a learned `W` can put mass on the cue regardless of position.
`examples/05_toy_attention.py` plants a peak hidden state and checks that
`argmax(alpha)` recovers it, which is the toy version of that claim.
