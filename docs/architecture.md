# Architecture

The project compares a frozen-embedding BiLSTM + attention model to
four sklearn baselines, each trained twice: **W** (Twitter GloVe only)
and **WE** (GloVe mean concatenated with emoji2vec mean).

## Deep model (`dl_model.PrepModel`)

```
Embedding(count, 200, weights=embedding_matrix, trainable=False)
  → Dropout(0.25)
  → Bidirectional(LSTM(256, return_sequences=True,
                       activation=tanh,
                       recurrent_activation=sigmoid,
                       kernel_initializer=he_normal))
  → Dropout(0.4)
  → Bidirectional(LSTM(256, return_sequences=True, …))
  → Dropout(0.4)
  → Attention()          # Raffel et al. 2015
  → Dense(1, sigmoid)
```

Compiled with `Adam(lr=0.001)`, `binary_crossentropy`, metric `acc`.

The saved Keras summaries in
`evaluate_loaded_dl_models.ipynb` report **2,510,848** parameters and
a sequence length of **78**. The embedding weights are wrapped as
`ModuleWrapper` because the model was saved from a mixed
`tensorflow.keras` / `tensorflow.python.keras` import path — see
`dl_model.py`.

### Why bidirectional + attention

A tweet is short, but sarcasm often sits in a late cue (`#not`, `😒`)
that should re-weight an earlier positive phrase (`I love walking to
school`). The first BiLSTM reads left-to-right *and* right-to-left
with `return_sequences=True`, so every step still has a 512-d hidden
state (2 × 256). Attention then does a soft selection over those
steps instead of trusting the final state alone.

### Attention math

`attention_layer.py` is the Keras 2 layer. For a batch of hidden
states `x` with shape `(samples, steps, features)`:

```
e_t = tanh(x_t · W + b_t)
a_t = softmax_t(e_t)          # mask applied after exp, then ε in the denominator
h   = Σ_t a_t x_t
```

`W` is a vector of length `features`. `b` is a per-step bias of
length `steps` (tied to the *training* `maxlen`, which is why the
layer is awkward to reuse at a different length). Masking is
supported (`supports_masking = True`) but the layer then returns
`compute_mask → None`, so nothing downstream sees the pad mask.

`python3 examples/attention_numpy.py` runs the same equations on a
6-step toy sequence and on a tokenized tweet. The last two toy steps
are scaled up so the weight bar is visible; a mask on the final step
shows the renormalization that the Keras `epsilon` term is there for.

### Single-modal vs multi-modal deep models

The two saved runs (`model/best_model_single_modal` and
`model/best_model_multi_modal`) share the architecture above. The
only intended difference is the **embedding matrix**:

* single-modal (`get_emoji2vec=False`): unknown tokens, including
  emoji, stay at zero
* multi-modal (`get_emoji2vec=True`): emoji characters are averaged
  from `emoji2vec_twitter.bin` before the LSTM ever sees them

Both matrices are frozen (`trainable=False` in `PrepModel`). Any
gain on the WE run is therefore from better *initial* token
geometry, not from fine-tuning emoji rows.

## Classical baselines

`baseline_models.ipynb` mean-pools each tweet (see
[preprocessing.md](preprocessing.md)) and fits:

| Estimator | sklearn class | Notes from the notebook |
| --- | --- | --- |
| SVM | `SVC()` | defaults (RBF). Both W and WE were actually trained as SVC. |
| Decision tree | `DecisionTreeClassifier()` | WE training cell accidentally constructs `SVC()` if the pickle is missing |
| Random forest | `RandomForestClassifier()` | W training cell accidentally constructs `SVC()` if the pickle is missing |
| Gradient boosting | `GradientBoostingClassifier()` | W training cell accidentally constructs `SVC()` if the pickle is missing |

Those three "accidental SVC" branches only run when the pickle is
absent. The checked-in `baseline_models/*.pkl` files are DT and GBT
only; SVM and RF pickles were never pushed. Treat the *recorded*
metrics in [results.md](results.md) as the source of truth, not a
re-fit from the except-blocks.

Input dimensionalities:

* W: 200
* WE: 400 (word mean ∥ emoji mean)

## Rule baseline (new, for the docs)

`examples/sarcasm_cues.py` predicts sarcastic if the tweet has an
explicit marker (`#not`, `#yeahright`, `#sarcastictweet`, …) or a
positive lexicon word plus a groan emoji. It is not part of the 2023
grade; it is a lower bound so the 87% number has something honest to
beat. Precision is high, recall is not — most sarcastic test tweets
have no such marker.

## What was *not* tried

The course write-up did not include transformers, character CNNs,
or learned emoji embeddings. The point was a controlled W vs WE
comparison on one dump, not a leaderboard. If you extend the project,
keep that pair of runs so a new model has the same two columns as
the tables in `examples/lib/metrics.py`.
