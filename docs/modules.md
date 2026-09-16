# Module map

How the original course files relate to the stdlib toolkit used by
`examples/`.

## Original (notebooks)

| File | Role |
| --- | --- |
| `data_utils.py` | `ReadOpen`, mean GloVe / emoji2vec, Keras `Preprocess` |
| `attention_layer.py` | Keras `Attention` (Raffel, mask-aware) |
| `dl_model.py` | `PrepModel`: Embedding → dropout → 2× BiLSTM → Attention → sigmoid |
| `baseline_models.ipynb` | SVM / DT / RF / GBT on mean vectors |
| `evaluate_loaded_dl_models.ipynb` | `load_model` + `evaluate` |
| `get_metrics_of_models.ipynb` | sklearn accuracy / F1 / P / R + bar chart |

`data_utils.AverageVectorPerTweet` skips out-of-vocab tokens and returns
a zero vector when a tweet has none. `Preprocess` writes those OOV rows
into the embedding matrix instead, optionally filling emoji from
emoji2vec.

## Toolkit (examples)

| Module | Replaces / explains |
| --- | --- |
| `sarcasm_toolkit.dataset` | `ReadOpen` without pandas; keeps raw text |
| `sarcasm_toolkit.tokenize` | NLTK `TweetTokenizer` stand-in (not identical) |
| `sarcasm_toolkit.cues` | surface features the notebooks never used |
| `sarcasm_toolkit.attention` | `Attention.call` in pure Python |
| `sarcasm_toolkit.embeddings` | 8-d toy table instead of GloVe |
| `sarcasm_toolkit.metrics` | sklearn `accuracy_score` / `f1_score` / … |
| `sarcasm_toolkit.baseline` | lexicon + logistic (not SVM/RF) |
| `sarcasm_toolkit.results` | loader for `examples/reported_results.json` |
| `sarcasm_toolkit.paths` | `dataset/`, `examples/`, `model/` locations |

Public imports (see `sarcasm_toolkit/__init__.py`):

```python
from sarcasm_toolkit import (
    load_split,
    tokenize_tweet,
    extract_cue_features,
    LexiconBaseline,
    CueLogistic,
    attention_pool,
    binary_metrics,
)
```

`python -m sarcasm_toolkit` prints split summaries and the 2023 score
table.

## Intentionally not wrapped

* Keras `Embedding` / `Bidirectional` / `LSTM`
* Gensim `KeyedVectors.load_word2vec_format`
* `joblib.load` of the 2023 pickles
* Any HTTP client

Those stay in the notebooks so the examples keep running on a stock
Python 3.12 with zero pip packages (tests use the stdlib `unittest`
runner).
