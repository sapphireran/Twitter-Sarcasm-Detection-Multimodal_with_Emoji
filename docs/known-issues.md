# Known issues

Recorded so the docs do not pretend the 2023 upload is a clean package.

## Fallback training fits the wrong estimator

In `baseline_models.ipynb`:

| Cell | Intended multimodal / single fit | Actual fallback |
| --- | --- | --- |
| Decision tree | `DecisionTreeClassifier()` on both views | multimodal branch is `SVC()` |
| Random forest | `RandomForestClassifier()` | single-modal branch is `SVC()` |
| Gradient boosting | `GradientBoostingClassifier()` | single-modal branch is `SVC()` |

SVM's fallback is the only one that constructs `SVC()` on purpose.

The executed outputs say "Loaded … models from files successfully", so the published table is not this bug. A fresh clone *will* hit it for SVM and RF because those pickles are not in git.

## Inconsistent pickle names

`baseline_models.ipynb` writes `svm_classifier.pkl`.
`get_metrics_of_models.ipynb` reads `svm_model.pkl`.

Same split for the `*_we` files. DT / GBT happen to use `*_classifier.pkl` in both.

## CSV paths

Two notebooks open `train_sentence.csv`. The files live in `dataset/`. Only the metrics notebook already uses the subdirectory.

## Gensim 4

`if j in model_word2vec.vocab` raises on Gensim 4. Use `j in model_word2vec` (or `j in model_word2vec.key_to_index`).

## Keras import mix

`dl_model.py` imports:

```python
from tensorflow.keras.layers import LSTM, Dropout, ...
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import *
from tensorflow.keras.optimizers import Adam
...
Adam(lr=lrate)
```

`tensorflow.python.keras` is gone from current TF. `Adam(lr=)` is `learning_rate=`. Star-importing `layers.core` is how `Dropout` / `Dense` sometimes resolve to a different package than `LSTM`.

## Embedding table width is `count`, not `vocab + 1`

`Preprocess` allocates `zeros((count, 200))` with `count = n_training_tweets`. It works because Keras word indices are `1..vocab_size` and `vocab_size < n_tweets`. It is still the wrong formula and wastes rows 0 and `vocab+1 … count-1`.

## Attention bias is length `steps`, not `features`

`b` has shape `(input_shape[1],)` — the padded length. Change `maxlen` and a loaded layer with a stored `b` of length 78 will not deserialize. This is why the saved models are glued to length 78.

## SavedModel completeness

`model/best_model_*/` contains `saved_model.pb` and `keras_metadata.pb` only. A TF SavedModel usually also has a `variables/` directory. `load_model` may fail even on a matching TF version.

## `#not` is not a gold label

Training contains non-sarcastic tweets that still include `#Not` / `#not ready`. A rule-based hashtag detector would look strong on the subtest and dirty on train. The models are right to use more than the hashtag.

## Shuffle vs no-shuffle

`ml_read_data` shuffles. `ReadOpen` + `Preprocess` does not. Do not reuse `y` from one path with `X` from the other.

## GloVe is not in the repository

Expected. The file is huge. The examples package exists so the rest of the write-up is still runnable.

## This docs pass does not change experiment code

The user request for this branch is personal documentation and examples. The bugs above are documented, not silently patched, so the 2023 notebook outputs stay the source of the result table.
