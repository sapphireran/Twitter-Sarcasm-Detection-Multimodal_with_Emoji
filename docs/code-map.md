# Code map

Personal research code only. Each module below is what the 2023
notebooks actually import.

## `data_utils.py`

Four public functions and two helpers.

### `ReadOpen(filename, Labelfile)`

Returns `(data, labels, count)` where `data` is a list of token lists
and `labels` is a 1-d integer array.

Quirk: commas inside a tweet are treated as delimiters and then
rejoined with spaces. That is why a quoted CSV field such as

```
"So many useless classes , great to be student"
```

becomes the token stream `so many useless classes great to be student`
after `TweetTokenizer` + lowercasing. The extra quotes disappear
because they sit on the comma-split pieces.

### `AverageVectorPerTweet` / `AverageVectorPerEmoji`

Identical loops, different KeyedVectors objects. For each tweet:

- collect in-vocabulary token vectors
- mean-pool on axis 0
- if the list is empty, emit `zeros(200)`

`AverageVectorPerEmoji` only contributes when the tokenizer left
emoji as their own tokens **and** those tokens exist in the emoji
model. Combined emoji such as ☺️ or skin-tone sequences may miss.

### `ml_read_data(...)`

Builds the classical (W, WE) matrices and shuffles them with one
shared index permutation. Used by `baseline_models.ipynb` and
`get_metrics_of_models.ipynb`.

### `Preprocess(docs, count, glove_model, emoji2vec_model, get_emoji2vec=True)`

Keras `Tokenizer` + `pad_sequences(..., padding='post')`. The
embedding matrix has `count` rows (number of **training tweets**, not
`len(word_index)+1`). That is a 2023 leftover: it happens to be large
enough for this vocabulary, but the correct size is
`len(tokenizer.word_index) + 1`. The examples walkthrough prints
both numbers so the mismatch is visible.

OOV handling:

1. Try the token in GloVe.
2. Else run `emoji.emoji_list(word)` and keep characters for which
   `emoji.is_emoji` is true.
3. Average those emoji2vec rows if `get_emoji2vec` is true.
4. On any failure, write a zero row and increment a silent `nf`
   counter (never returned).

`get_emoji2vec=False` is the single-modal (W) embedding table: OOV
emoji become zeros.

### `preprocess_test(tokenizer, maxlen, test_docs)`

Reuses the training tokenizer and pads / truncates to the training
`maxlen`.

## `attention_layer.py`

Keras `Layer` implementing Raffel-style feed-forward attention over a
time axis.

Input: `(batch, steps, features)`.
Output: `(batch, features)`.

Scoring:

```
e_t = tanh(x_t · W + b_t)
a   = softmax(e)          # mask applied after exp, + epsilon
h   = sum_t a_t * x_t
```

`W` is a vector of size `features` (not a matrix). `b` is a vector of
size `steps` if `bias=True`. That bias is **position-specific**, so
the layer stores a weight per timestep and is not length-agnostic
after `build()`. The saved models were built at length 78.

`compute_mask` returns `None`, so the collapsed vector is unmasked
for the dense head.

The examples file `examples/attention_demo.py` reimplements `call()`
in NumPy so you can see the weights on a toy 4-step sequence.

## `dl_model.py`

`PrepModel(count, embedding_matrix, l, lrate=0.001)` builds:

```
Embedding(count, 200, weights=embedding_matrix, trainable=False)
Dropout(0.25)
Bidirectional(LSTM(256, return_sequences=True,
                   kernel_initializer='he_normal',
                   activation='tanh',
                   recurrent_activation='sigmoid'))
Dropout(0.4)
Bidirectional(LSTM(256, return_sequences=True, ...))
Dropout(0.4)
Attention()
Dense(1, activation='sigmoid')
```

Optimizer: `Adam(lr=lrate)`. Loss: `binary_crossentropy`. Metric:
`acc`.

Two stacked bidirectional LSTMs with `return_sequences=True` produce
a `(batch, 78, 512)` tensor (256 × 2). Attention pools that down to
`(batch, 512)` before the sigmoid.

The saved `model/best_model_*` summaries in
`evaluate_loaded_dl_models.ipynb` match this: 2,510,848 parameters,
all marked trainable in the loaded graph (the frozen embedding flag
from `PrepModel` is not always preserved across the 2023 export).

## Notebooks

### `baseline_models.ipynb`

Loads `glove.twitter.27B.200d.bin` and `emoji2vec_twitter.bin` from
the **repo root**, then calls `ml_read_data` with filenames that
assume the CSVs are also in the root (`train_sentence.csv`, not
`dataset/train_sentence.csv`). If you re-run it, either symlink the
files or edit the paths.

For each of SVM, Decision Tree, Random Forest, and Gradient Boosting
it tries to `joblib.load` a pair of pickles (`*_classifier.pkl` and
`*_classifier_we.pkl`). On `FileNotFoundError` it fits defaults
(`SVC()`, `DecisionTreeClassifier()`, …) and dumps them.

Recorded full-test accuracies from that notebook:

| Model | W | WE |
| --- | ---: | ---: |
| SVM | 0.769 | 0.763 |
| Decision Tree | 0.7265 | 0.7295 |
| Random Forest | 0.8145 | 0.818 |
| Gradient Boosting | 0.746 | 0.7475 |

This checkout only has the Decision Tree and Gradient Boosting
pickles under `baseline_models/`. SVM and Random Forest pickles were
not uploaded.

### `evaluate_loaded_dl_models.ipynb`

Rebuilds the Keras sequences with `Preprocess` / `preprocess_test`,
then `tf.keras.models.load_model` on:

- `model/best_model_multi_modal`
- `model/best_model_single_modal`

Recorded `evaluate()` accuracies:

| Model | Test acc | Subtest acc |
| --- | ---: | ---: |
| WE (multi) | 0.8735 | 0.8921 |
| W (single) | 0.8635 | 0.8669 |

The SavedModel directories here contain `saved_model.pb` and
`keras_metadata.pb` but no `variables/` shard. Reloading them in a
modern TensorFlow will likely fail until the weight files are
restored. Treat the notebook outputs as the historical result.

### `get_metrics_of_models.ipynb`

Same loaders, plus F1 / recall / precision and the comparison plot.
Full tables are copied into [results.md](results.md).

## Binary assets

| File | Used as | Notes |
| --- | --- | --- |
| `emoji2vec_twitter.bin` | emoji channel in every notebook | Present (~1.3 MB) |
| `emoji2vec.bin` | unused by the notebooks | Present (~2.0 MB); generic Eisner vectors |
| `glove.twitter.27B.200d.bin` | word channel | **Not in git** |
| `glove_tt.txt` | alternate text GloVe in `get_metrics` | **Not in git** |

Download GloVe Twitter 27B from the Stanford GloVe page and convert
it with Gensim if you want to replay training. See
[reproduction.md](reproduction.md).
