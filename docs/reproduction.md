# Reproduction notes

This page is for a future you (or anyone else) who wants to replay
the 2023 notebooks. The `examples/` scripts do not need any of this.

## What is already in git

- Sentence / label CSVs for train, test, and subtest
- `emoji2vec_twitter.bin` and `emoji2vec.bin`
- `attention_layer.py`, `data_utils.py`, `dl_model.py`
- Three result notebooks with stored outputs
- Decision Tree and Gradient Boosting pickles
- Keras SavedModel *graphs* under `model/best_model_*`

## What is not in git

| Artifact | Why it matters | Where to get it |
| --- | --- | --- |
| `glove.twitter.27B.200d.bin` | Word channel for every W/WE run | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) Twitter 27B, then convert with Gensim |
| `glove_tt.txt` | Text-format GloVe used in `get_metrics_of_models.ipynb` | Same file, skip binary conversion |
| `baseline_models/svm_*.pkl` | Recorded SVM numbers | Retrain with `SVC()` |
| `baseline_models/rf_*.pkl` | Recorded RF numbers | Retrain with `RandomForestClassifier()` |
| `model/*/variables/*` | Deep weights | Not recoverable from `saved_model.pb` alone |

GloVe Twitter 27B is 1.42 GB compressed for the full set of
dimensions. You only need the 200-d vectors.

### Convert GloVe to word2vec binary

```bash
python - <<'PY'
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
glove2word2vec("glove.twitter.27B.200d.txt", "glove.twitter.27B.200d.w2v.txt")
kv = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.w2v.txt", binary=False)
kv.save_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
PY
```

Put the `.bin` next to the notebooks (repo root). The metrics
notebook instead loads `glove_tt.txt` as `binary=False` — that is
the same 200-d table in text form.

## Python stack the notebooks expect

The code was written against **TensorFlow / Keras 2.x** and
`keras_preprocessing` as a standalone package. A modern
`tensorflow>=2.16` install will not import:

```python
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import *
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
```

`dl_model.py` also uses `Adam(lr=lrate)` (`lr=` was renamed to
`learning_rate`).

A conservative environment that matches the source more closely:

```
python 3.8 or 3.9
tensorflow==2.10
keras-preprocessing
gensim==3.8.3          # .vocab on KeyedVectors; gensim 4 moved this to .key_to_index
nltk
emoji
pandas
numpy
scikit-learn
joblib
matplotlib
```

Gensim 4 will raise on `model_word2vec.vocab` in
`AverageVectorPerTweet`. Either pin Gensim 3 or change those
membership tests to `word in model_word2vec`.

NLTK needs the tweet tokenizer tables once:

```python
import nltk
nltk.download("punkt")
```

`TweetTokenizer` itself ships with the `nltk` package and does not
need an extra corpus for basic use.

## Path mismatches

| Notebook | Looks for CSVs in | Looks for GloVe in |
| --- | --- | --- |
| `baseline_models.ipynb` | repo root (`train_sentence.csv`) | `glove.twitter.27B.200d.bin` |
| `evaluate_loaded_dl_models.ipynb` | repo root | `glove.twitter.27B.200d.bin` |
| `get_metrics_of_models.ipynb` | `dataset/` | `glove_tt.txt` |

Easiest fix: symlink the dataset files into the repo root.

```bash
ln -s dataset/train_sentence.csv .
ln -s dataset/train_label.csv .
ln -s dataset/test_sentence.csv .
ln -s dataset/test_label.csv .
ln -s dataset/subtest_sentence.csv .
ln -s dataset/subtest_label.csv .
```

## Retraining the deep model

There is no committed training loop — only `PrepModel` and the
evaluation notebooks. A minimal fit that matches the builder:

```python
from dl_model import PrepModel
from data_utils import ReadOpen, Preprocess, preprocess_test

data, y, n = ReadOpen("dataset/train_sentence.csv", "dataset/train_label.csv")
X, emb, L, tok = Preprocess(data, n, glove, emoji2vec, get_emoji2vec=True)
model = PrepModel(n, emb, L, lrate=0.001)
model.fit(X, y, batch_size=32, epochs=5, validation_split=0.1)
```

That is **not** a claim about the original epoch count, batch size,
or early stopping. Those were never checked in. Treat any new weights
as a new experiment.

## Evaluating without TensorFlow

Use the example scripts:

```bash
python examples/inspect_dataset.py
python examples/preprocess_walkthrough.py
python examples/emoji_signal.py
python examples/heuristic_baseline.py
python examples/attention_demo.py
python examples/report_metrics.py
```

They only need the Python standard library plus NumPy (already used
by the original project).

## Known bugs to fix before a retrain

1. `baseline_models.ipynb` fallback cells fit `SVC()` for several
   "W" models that are labeled Decision Tree / Random Forest /
   Gradient Boosting.
2. `Preprocess` sizes the embedding matrix to `count` (tweet count)
   instead of `len(tokenizer.word_index) + 1`.
3. `AverageVectorPerTweet` uses `model.vocab` (Gensim 3 only).
4. Bare `except:` in `Preprocess` swallows missing emoji2vec keys.
5. `np.random.permutation` in `ml_read_data` is unseeded and is
   applied to **test** as well as train.

None of these were changed in the docs/examples expansion. The
examples restate them so a later cleanup can be deliberate.
