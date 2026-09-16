# Architecture

Two families of models share the same embeddings and the same labels.

```
                    ┌─ AverageVectorPerTweet (200-d) ─────────┐
 tweet line         │                                         ├─ SVM / DT / RF / GBT
 ReadOpen tokens ───┤                                         │
                    └─ concat(word avg, emoji avg) (400-d) ───┘

                    ┌─ GloVe-only embedding matrix ───────────┐
 same tokens        │  Embedding → Drop → BiLSTM → Drop       │
 Keras Tokenizer ───┤  → BiLSTM → Drop → Attention → Dense    ├─ single-modal BiLSTM
                    │                                         │
                    └─ GloVe + emoji2vec fallback matrix ─────┘  multi-modal BiLSTM
```

The classical path **destroys order** (mean pool). The deep path **keeps order** (padded sequences, max length taken from the longest train tweet after tokenization — 78 in the saved-model summary).

## Tokenization

`data_utils.ReadOpen`:

1. Read the sentence file as UTF-8 (`errors="replace"`).
2. `sentence = ' '.join(line.strip().split(','))` — flatten comma-separated leftovers.
3. `nltk.TweetTokenizer().tokenize(sentence)` then `.lower()` every token.
4. Load labels with pandas and `squeeze()` to a 1-d array.

Deep models then run `keras_preprocessing.text.Tokenizer.fit_on_texts` on those token lists and `pad_sequences(..., padding='post')`. Test / subtest reuse the **train** tokenizer via `preprocess_test`.

That means evaluation vocab is frozen on train. A test-only emoji still gets an integer id only if the Keras tokenizer saw that exact string in train (or it becomes OOV index 0 / dropped depending on Keras settings — default Tokenizer drops OOV). Combined with the emoji2vec *matrix* fallback, an emoji that never appeared in train cannot enter the sequence at all. Subtest performance is therefore “emoji the model has already seen as tokens,” not “any emoji at all.”

## Embedding matrix (`Preprocess`)

For each `word, i` in `tokenizer.word_index`:

1. If `word` is in the GloVe `KeyedVectors` vocab, copy the 200-d row.
2. Else collect emoji codepoints inside the token (`emoji.emoji_list` + `emoji.is_emoji`).
3. If any, average their emoji2vec rows.
4. If `get_emoji2vec=False` (single-modal ablation), write zeros instead of that average.
5. On any lookup failure, write zeros and increment a miss counter.

The embedding layer is created with `trainable=False`. The network is not allowed to move GloVe / emoji2vec; it only learns the recurrent + attention + dense weights (~2.51M trainable params in the saved summaries).

Classical `AverageVectorPerTweet` / `AverageVectorPerEmoji` do the same lookup per token and mean-pool. Empty tweets become a zero vector. Multi-modal classical features are `concat(word_avg, emoji_avg)` → 400-d. Tokens that are not emoji contribute **zeros** to the emoji half, so the second block is sparse on text-only tweets.

## Deep classifier (`dl_model.PrepModel`)

| Layer | Config | Why |
| --- | --- | --- |
| `Embedding` | `count × 200`, frozen, `input_length=l` | Shared space for words and emoji |
| `Dropout` | 0.25 | Regularize the embedding stream |
| `Bidirectional(LSTM 256)` | `he_normal`, `tanh` / `sigmoid` recurrent, `return_sequences=True` | Forward + backward context → 512-d per step |
| `Dropout` | 0.40 | |
| `Bidirectional(LSTM 256)` | same, `return_sequences=True` | Deeper temporal features |
| `Dropout` | 0.40 | |
| `Attention` | Raffel et al. 2015 | Soft-select steps (hashtags, emoji, punchline) |
| `Dense(1, sigmoid)` | | Binary sarcasm |
| Optim | `Adam(lr=0.001)` | `binary_crossentropy`, metric `acc` |

Saved-model `.summary()` printouts show `module_wrapper_*` names because the 2023 export wrapped standalone Keras layers through `tensorflow.python.keras` compatibility shims. Shapes still read `(None, 78, 200)` → `(None, 78, 512)` → `(None, 512)` → `(None, 1)`.

## Attention (`attention_layer.Attention`)

Input `x` has shape `(batch, steps, features)`.

```
e_t = tanh( x_t · W + b_t )     # W: (features,), b: (steps,) if bias
α_t = softmax_masked(e)         # ε added to the denominator
h   = Σ_t α_t x_t               # (batch, features)
```

This is the feed-forward attention from [Raffel et al., “Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems”](https://arxiv.org/abs/1512.08756). It supports masking: masked steps have their `exp(e)` zeroed before the normalize.

Two implementation notes that bite at export / reuse time:

- `b` is **per-step**, not per-feature. The layer stores a vector of length `input_shape[1]` (the padded length). A model built at length 78 cannot be reloaded onto a different pad length without rebuilding `b`.
- `compute_mask` returns `None`, so downstream layers do not see the original padding mask. Padding must already have been handled inside `call`.

`examples/attention_demo.py` is a NumPy clone of `call()` with the same ε trick, used in tests as a specification.

## Classical models

Notebooks fit four sklearn estimators, each twice (word-only 200-d and word+emoji 400-d):

| Estimator | Default-ish constructor in the notebook |
| --- | --- |
| `SVC()` | RBF SVM |
| `DecisionTreeClassifier()` | Unpruned tree |
| `RandomForestClassifier()` | Default 100 trees |
| `GradientBoostingClassifier()` | Default GBT |

`ml_read_data` **shuffles train and test with `np.random.permutation`** after embedding. Labels stay aligned (same index permutation). There is no fixed seed in the function, so re-running feature extraction without saving the arrays will not reproduce row order. Metrics are computed on the already-embedded arrays, so this only matters if you re-extract.

Pickles present in git today:

- `baseline_models/dt_classifier.pkl` + `dt_classifier_we.pkl`
- `baseline_models/gbt_classifier.pkl` + `gbt_classifier_we.pkl`

SVM and Random Forest pickles were referenced by the notebooks (`svm_model.pkl`, `rf_classifier.pkl`, …) but were not uploaded. Accuracy numbers for those two still exist in notebook stdout.

## File map

| File | Responsibility |
| --- | --- |
| `data_utils.py` | I/O, averages, Keras tokenizer + embedding matrix |
| `attention_layer.py` | Custom Keras attention |
| `dl_model.py` | Sequential factory |
| `baseline_models.ipynb` | Train or load sklearn models, print accuracies |
| `evaluate_loaded_dl_models.ipynb` | `tf.keras.models.load_model` on both SavedModels |
| `get_metrics_of_models.ipynb` | Acc / F1 for everything, plus a comparison plot |
| `examples/lib/` | Dependency-light re-implementations for docs |
