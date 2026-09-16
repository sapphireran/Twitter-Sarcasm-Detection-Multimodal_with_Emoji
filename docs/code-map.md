# Code map

A file-level tour for anyone opening this repo after the course.

## Runtime graph (original)

```
dataset/*.csv
        │
        ▼
 data_utils.ReadOpen          ─► list[list[str]], np.ndarray labels
        │
        ├─ ml_read_data       ─► X (N,200), X_emoji (N,400)     ─► sklearn
        │
        └─ Preprocess         ─► padded ids, emb (V,200), maxlen
                │
                ▼
         dl_model.PrepModel   ─► keras Sequential
                │
                ▼
         attention_layer.Attention
```

## `data_utils.py`

| Function | Input | Output |
| --- | --- | --- |
| `ReadOpen(filename, Labelfile)` | two paths | `data, labels, len(lines)` |
| `AverageVectorPerTweet(data, glove)` | token lists, KeyedVectors | list of 200-d means |
| `AverageVectorPerEmoji(data, e2v)` | token lists, KeyedVectors | list of 200-d means |
| `ml_read_data(...)` | paths + two models | shuffled `X, y, X_emoji, y_emoji` |
| `Preprocess(docs, count, glove, e2v, get_emoji2vec=True)` | token lists | `padded, matrix, maxlen, tokenizer` |
| `preprocess_test(tokenizer, maxlen, test_docs)` | frozen tokenizer | padded test ids |

`count` is `len(lines)` from `ReadOpen`, used as the embedding table height (`zeros((count, 200))`). Keras word indices start at 1, so row 0 stays zero (pad / OOV). If the number of unique tokens ever exceeded `count` (unique words > number of tweets) the matrix would be too short. With ~40k tweets and a short vocab this did not fire.

## `dl_model.py`

Single factory. Imports both `tensorflow.keras.layers.Embedding` and `tensorflow.python.keras.layers.embeddings.Embedding` — the second shadows nothing useful; `e = Embedding(...)` resolves to the first import. Safe to delete the python.keras lines when modernizing.

## `attention_layer.py`

Standalone `Layer` subclass. See [architecture.md](architecture.md) for the equations. The docstring still says “tested with Keras 2.0.6”; the 2023 notebooks ran a TF 2 install that wrapped it.

## Notebooks

| Notebook | Does |
| --- | --- |
| `baseline_models.ipynb` | Load GloVe + emoji2vec, `ml_read_data`, load-or-fit four sklearn pairs, print acc |
| `evaluate_loaded_dl_models.ipynb` | `Preprocess` + `load_model` both SavedModels, `evaluate`, `summary` |
| `get_metrics_of_models.ipynb` | Acc + F1 for sklearn and both BiLSTMs, matplotlib comparison |

Paths are inconsistent across notebooks (`train_sentence.csv` vs `dataset/train_sentence.csv`, `glove.twitter.27B.200d.bin` vs `glove_tt.txt`). Run them from a working directory where those relative paths resolve, or edit the first cells.

## `examples/` (this PR)

Dependency-light mirrors, not drop-in replacements:

| Module | Mirrors |
| --- | --- |
| `examples/lib/tweet_tokenize.py` | NLTK `TweetTokenizer` subset |
| `examples/lib/dataset_io.py` | `ReadOpen` without pandas / nltk |
| `examples/lib/emoji_extract.py` | the emoji walk inside `Preprocess` |
| `examples/lib/mean_pool.py` | `AverageVectorPerTweet` / concat |
| `examples/lib/attention_numpy.py` | `Attention.call` |
| `examples/lib/lexical_features.py` | cues the embeddings were free to exploit |
| `examples/lib/logistic.py` | a tiny binary logistic for the lexical floor |

Scripts under `examples/*.py` are CLIs over those modules. Tests import the library directly.

## Binaries

| File | What it is |
| --- | --- |
| `emoji2vec.bin` | Generic emoji2vec word2vec binary |
| `emoji2vec_twitter.bin` | Twitter-tuned sibling used by the notebooks |
| `baseline_models/*.pkl` | sklearn trees / GBTs |
| `model/best_model_*/saved_model.pb` | TF SavedModel graph (weights not uploaded) |

Do not commit new GloVe dumps; they are ~1–2 GB.
