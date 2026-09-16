# 05 — Reproduction notes

This page is a personal lab notebook for “I want to run the 2023 code
again.” The example scripts under `examples/` are the path that still
works in a numpy-only environment. Everything below is about the
notebooks and `data_utils.py`.

## What git actually contains

Present:

* CSVs under `dataset/`
* `emoji2vec.bin`, `emoji2vec_twitter.bin`
* `attention_layer.py`, `data_utils.py`, `dl_model.py`
* DT and GBT pickles in `baseline_models/`
* `model/best_model_{single,multi}_modal/keras_metadata.pb` only

Missing from git (needed for the original notebooks):

* `glove.twitter.27B.200d.bin` (or `glove_tt.txt` as in one notebook cell)
* SVM / RF pickles (`svm_classifier.pkl`, `rf_classifier.pkl`, `*_we.pkl`)
* SavedModel weight shards (`variables.data-*`, `variables.index`)
* The `emoji` package’s data files at the 2023 version

Without those, you can still inspect the CSVs and emoji2vec. You cannot
reproduce the 87% figure.

## Environment mismatches that will bite

### Gensim 4

```python
if j in model_word2vec.vocab:   # Gensim 3
    row.append(model_word2vec[j])
```

Gensim 4: use `token in model` or `token in model.key_to_index`.
`KeyedVectors.load_word2vec_format` itself still exists.

### Keras 2 vs 3, TF 2.x vs 2.16+

* `Adam(lr=0.001)` → `learning_rate`.
* `from tensorflow.python.keras.layers.embeddings import Embedding` and
  `from tensorflow.python.keras.layers.core import *` are private imports
  that already look wrong in TF 2.10 and are gone later. `dl_model.py`
  also imports `Embedding` twice. A port should use
  `tensorflow.keras.layers`.
* Custom `Attention` without `get_config` / `from_config` will not
  deserialize cleanly. Pass
  `custom_objects={"Attention": Attention}` and consider adding
  `get_config` before re-exporting.
* `keras_preprocessing` is a separate package in TF 2.9+; the original
  `from keras_preprocessing.text import Tokenizer` still works if that
  wheel is installed.

### NLTK

`TweetTokenizer` is in `nltk.tokenize.casual` and does not need `punkt`.
A missing NLTK install is a hard failure in `ReadOpen`.

### Pandas / label squeeze

`labels_pd.values.squeeze()` on a one-column frame yields a 1-d array of
ints if the CSV parsed as integers. A stray BOM or header would break
that; these files do not have a header.

### Paths

| Notebook | CSV path assumed |
| --- | --- |
| `baseline_models.ipynb` | working directory (`train_sentence.csv`) |
| `evaluate_loaded_dl_models.ipynb` | working directory |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` |

Either copy the CSVs up one folder or edit the paths.

## GloVe

Pennington et al. ship `glove.twitter.27B.200d.txt` as *text*. The
notebooks load a **binary** word2vec file. Conversion, once, with Gensim:

```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
glove2word2vec("glove.twitter.27B.200d.txt", "glove.twitter.27B.200d.w2v.txt")
kv = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.w2v.txt", binary=False)
kv.save_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
```

Do not commit that file; it is hundreds of megabytes.

## Suggested modern re-run (if you only want the claim)

1. Keep the CSVs and `emoji2vec_twitter.bin`.
2. Do **not** fight SavedModel. Re-declare `PrepModel`, build the embedding
   matrix with a current Gensim, train two seeds of W vs WE, evaluate on
   `test` and on the emoji-filtered subset (rebuild subtest as
   “test rows where `extract_emojis(text)` is non-empty” — it should be
   278).
3. Report mean ± std over seeds. The 2023 table is a single seed.

## Example-layer reproduction (this PR)

These are deterministic given the CSVs and the two `.bin` files:

```bash
python3 examples/01_explore_dataset.py
python3 examples/02_emoji_cooccurrence.py
python3 examples/03_lexical_cues.py
python3 examples/04_inspect_emoji2vec.py
python3 examples/05_toy_attention.py
python3 examples/06_bow_baseline.py
python3 -m unittest discover -s tests -v
```

`06_bow_baseline.py` is a control, not a reproduction of SVM/RF.
