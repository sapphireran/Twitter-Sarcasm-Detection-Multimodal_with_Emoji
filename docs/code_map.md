# Code map

A file-by-file guide to the original 2023 tree and the docs/examples additions.

## Original Python modules

### `data_utils.py`

| Function | Input | Output |
| --- | --- | --- |
| `ReadOpen(filename, Labelfile)` | two paths | `(list[list[str]], np.ndarray, int)` token lists, labels, line count |
| `AverageVectorPerTweet` | tokens + GloVe | list of 200-d averages |
| `AverageVectorPerEmoji` | tokens + emoji2vec | list of 200-d averages |
| `ml_read_data` | paths + both models | shuffled `X, y, X_emoji, y_emoji` |
| `Preprocess` | token lists + both models | padded ids, embedding matrix, maxlen, fitted Tokenizer |
| `preprocess_test` | fitted Tokenizer + maxlen + docs | padded test ids |

Side effects: `ml_read_data` shuffles with an unseeded permutation. `Preprocess` sizes the embedding table by tweet count, not vocab size.

### `attention_layer.py`

Keras `Layer` subclass. `supports_masking = True`. `compute_mask` returns `None` so the mask does not leak into `Dense`. Math is documented in [methodology.md](methodology.md). The header cites Keras 2.0.6; the notebooks ran on a later TensorFlow 2 install.

### `dl_model.py`

`PrepModel(count, embedding_matrix, l, lrate=0.001)` returns a compiled `Sequential`. Unused imports (`Flatten`, a second `Embedding` from `tensorflow.python.keras`) are leftovers.

## Original notebooks

### `baseline_models.ipynb`

Loads GloVe + emoji2vec, builds averaged features, then for each of SVM / DT / RF / GBT tries to `joblib.load` a pair of pickles and otherwise fits `sklearn` defaults. Prints four accuracies per family. Early cells assume the CSV files sit in the working directory, not under `dataset/`.

### `evaluate_loaded_dl_models.ipynb`

Rebuilds the Keras integer sequences from train (to fit the tokenizer) and evaluates the two SavedModel directories. Recorded accuracies: 0.8735 / 0.8921 (multimodal test / subtest) and 0.8635 / 0.8669 (text-only).

### `get_metrics_of_models.ipynb`

The grading / writeup notebook. Loads GloVe from `glove_tt.txt`, scores sklearn + Keras models on accuracy / F1 / precision / recall, and draws the comparison figure. Contains a large inline PNG.

## Serialized models

```
model/best_model_single_modal/   # text-only BiLSTM+ATT
model/best_model_multi_modal/    # GloVe + emoji2vec rows
baseline_models/dt_classifier.pkl
baseline_models/dt_classifier_we.pkl
baseline_models/gbt_classifier.pkl
baseline_models/gbt_classifier_we.pkl
```

`_we` means “with emoji” (400-d concatenated averages for sklearn). SavedModel folders are TensorFlow 2 SavedModels (`saved_model.pb` + `keras_metadata.pb`). **This checkout has no `variables/` directory under either model**, so the uploaded graphs cannot be restored until those shards are added back. The recorded notebook scores still stand as a historical run.

## Embeddings on disk

| File | Size (this checkout) | Used as |
| --- | ---: | --- |
| `emoji2vec.bin` | ~2.0 MiB | generic emoji2vec (not referenced by the notebooks) |
| `emoji2vec_twitter.bin` | ~1.3 MiB | emoji rows in every notebook |

GloVe Twitter 200-d is external.

## Docs / examples tree

```
docs/dataset.md              # card, counts, top tags / emoji
docs/methodology.md          # W vs WE, architecture, quirks
docs/results.md              # transcribed notebook metrics
docs/reproduction.md         # how to re-run the 2023 stack
docs/code_map.md             # this file
examples/sarcasm_lib/        # importable helpers (stdlib + NumPy)
examples/inspect_dataset.py
examples/emoji_signals.py
examples/lexical_baseline.py
examples/attention_demo.py
examples/preprocess_walkthrough.py
tests/                       # unittest coverage for the helpers
```

`examples/sarcasm_lib` does not import `data_utils.py`, so it stays runnable when Keras / NLTK / Gensim are absent.
