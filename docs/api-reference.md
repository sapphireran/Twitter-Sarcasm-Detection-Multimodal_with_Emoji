# API reference

Personal project modules only. The `examples/` package is documented in [examples/README.md](../examples/README.md).

## `data_utils.py`

### `ReadOpen(filename, Labelfile) -> tuple[list, np.ndarray, int]`

Read a tweet file and a parallel label file.

- Each sentence line is `strip`ped, split on commas, re-joined with spaces, then passed through `nltk.TweetTokenizer` and lowercased.
- Labels: `pandas.read_csv(Labelfile, index_col=False, header=None)` then `squeeze`.
- Returns `(tokenized_tweets, labels, line_count)`.
- Opens the sentence file with `encoding="utf-8", errors="replace"`.

### `AverageVectorPerTweet(data, model_word2vec) -> list[list[float]]`

Mean GloVe vector per tweet.

- `data[i]` is a list of tokens.
- Tokens not in `model_word2vec.vocab` are skipped.
- Empty tweets become `zeros(200)`.
- Returns a Python list of 200-d lists (later cast to `np.array` by `ml_read_data`).

### `AverageVectorPerEmoji(data, model_emoji2vec) -> list[list[float]]`

Same loop, different keyed-vectors object. Empty emoji sets become `zeros(200)`.

### `ml_read_data(data_file, label_file, glove_model, emoji2vec_model) -> tuple`

Convenience wrapper used by the sklearn notebooks.

Returns `(X, y, X_emoji, y_emoji)` where:

- `X` has shape `(n, 200)`
- `X_emoji` has shape `(n, 400)`
- `y` and `y_emoji` are the same labels after the same permutation

Side effect: shuffles with `np.random.permutation` (unseeded).

### `Preprocess(docs, count, glove_model, emoji2vec_model, get_emoji2vec=True) -> tuple`

Build padded training ids and a frozen embedding table.

| Arg | Meaning |
| --- | --- |
| `docs` | output of `ReadOpen` (list of token lists) |
| `count` | `len(docs)`; used as `Embedding` input dimension |
| `get_emoji2vec` | if False, GloVe-OOV rows stay zero |

Returns `(padded_docs, embedding_matrix, maxlen, tokenizer)`.

OOV fill path uses `emoji.emoji_list` + `emoji.is_emoji` and averages every emoji2vec hit. Failures increment a local `nf` counter and write zeros. The counter is not returned.

### `preprocess_test(tokenizer, maxlen, test_docs) -> np.ndarray`

`texts_to_sequences` + `pad_sequences(..., maxlen=maxlen, padding='post')`.

## `dl_model.py`

### `PrepModel(count, embedding_matrix, l, lrate=0.001) -> keras.Model`

Sequential graph described in [models.md](models.md).

- `count` / `embedding_matrix` / `l` must match `Preprocess`.
- Embedding is created with `trainable=False`.
- Compiles `Adam(lr=lrate)`, `binary_crossentropy`, `metrics=['acc']`.

Imports mix `tensorflow.keras` and `tensorflow.python.keras`. That worked on the 2023 Colab / local TF the author used. It is the first thing that breaks on a current `pip install tensorflow`.

## `attention_layer.py`

### `class Attention(Layer)`

Raffel-style temporal attention.

Constructor flags: `W_regularizer`, `b_regularizer`, `W_constraint`, `b_constraint`, `bias=True`.

Methods:

| Method | Behavior |
| --- | --- |
| `build(input_shape)` | asserts 3-d input; adds `W` `(features,)` and optional `b` `(steps,)` |
| `call(x, mask=None)` | `e = tanh(xW + b)`, softmax with `epsilon`, weighted sum over steps |
| `compute_mask(...)` | always `None` |
| `compute_output_shape(...)` | `(batch, features)` |

`supports_masking = True`, so an upstream Embedding mask *can* zero out pad scores before the softmax.

## Notebooks (not importable APIs)

| File | Entry point |
| --- | --- |
| `baseline_models.ipynb` | load-or-fit four sklearn pairs, print accuracy |
| `evaluate_loaded_dl_models.ipynb` | `load_model` + `evaluate` + `summary` |
| `get_metrics_of_models.ipynb` | accuracy, F1, precision, recall, bar chart |

See [notebooks.md](notebooks.md).

## Examples package (lightweight)

Importable as `examples.*` if the repo root is on `PYTHONPATH` (the scripts add it themselves).

| Module | Public functions |
| --- | --- |
| `examples.tokenize` | `tweet_tokenize`, `read_sentence_file`, `read_label_file` |
| `examples.hash_embeddings` | `HashEmbeddings.embed`, `.contains` |
| `examples.average_vectors` | `average_channel`, `multimodal_features` |
| `examples.fusion` | `concat_channels`, `cosine`, `channel_norms` |
| `examples.attention_numpy` | `attention_forward`, `softmax` |
| `examples.toy_classifier` | `fit_logreg`, `predict_proba`, `metrics` |
| `examples.toy_corpus` | `TOY_TWEETS`, `TOY_LABELS`, `illustrative_pair` |
| `examples.dataset_stats` | `summarize_split`, `iter_pairs` |

These functions are the ones the unit tests lock down.
