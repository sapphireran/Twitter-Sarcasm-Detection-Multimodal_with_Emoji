# Reproduction notes

Two different "run this repo" stories exist. Do not mix them.

## Path A — examples and tests (this branch)

Needs Python 3.10+ and NumPy. TensorFlow, Gensim, NLTK, and GloVe are not
required.

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/01_dataset_preview.py
python3 examples/02_lexical_cues.py
python3 examples/03_emoji_vectors.py
python3 examples/04_attention_walkthrough.py
python3 examples/05_tfidf_baseline.py
python3 -m pytest tests/ -q
```

All scripts resolve the repository root from their own file location, so
they work from any working directory.

What you should see:

- preview: 39,780 / 2,000 / 278 rows and the class counts in
  [`dataset.md`](dataset.md)
- lexical cues: `#not` is almost perfectly sarcastic on the test set
- emoji vectors: header `1661 200`, a few cosine neighbours for 😂 and 😒
- attention: softmax rows sum to 1, masking moves the mass
- TF-IDF: test accuracy near 0.83 with hashtags and near 0.74 without
  (hash size 4,096, five epochs, seed 0)

`examples/03_emoji_vectors.py` reads `emoji2vec_twitter.bin` with
`examples/lib/word2vec_bin.py`. It does not import Gensim.

## Path B — 2023 notebooks (GloVe + TensorFlow)

This is the original course path. It is **blocked** in a fresh clone by
missing files.

### Missing pieces

| Artifact | Typical size | Used by |
| --- | --- | --- |
| `glove.twitter.27B.200d.bin` or `glove_tt.txt` | ~1.2 GB | every training/eval notebook |
| full Keras SavedModel trees under `model/` | tens of MB | `evaluate_loaded_dl_models.ipynb` |
| `baseline_models/*.pkl` | small | `baseline_models.ipynb` load branch |

This checkout only has `model/*/keras_metadata.pb`. That is not enough for
`tf.keras.models.load_model`.

GloVe Twitter 27B is distributed by the Stanford NLP group. The project
used the 200-d file converted to word2vec binary (`KeyedVectors.load_word2vec_format(..., binary=True)`). The metrics notebook instead loads a
text file named `glove_tt.txt`. Either format works if the path and
`binary=` flag match.

### Environment drift

The 2023 code was written against:

- `gensim.models.KeyedVectors` with `.vocab` (Gensim 3.x). Gensim 4
  removed `.vocab`; use `key_to_index` or `word in model`.
- `keras_preprocessing.text.Tokenizer` and
  `keras_preprocessing.sequence.pad_sequences`. Current TensorFlow still
  vendors these, but the import path may be
  `tensorflow.keras.preprocessing`.
- `Adam(lr=0.001)`. Current Keras wants `learning_rate`.
- `tensorflow.python.keras.layers.embeddings.Embedding` and
  `tensorflow.python.keras.layers.core` inside `dl_model.py`. Those
  private imports break on newer TF. Prefer
  `tensorflow.keras.layers.Embedding` (already imported on the line
  above — the private import is redundant).
- Custom `Attention` as a raw `Layer` subclass without `get_config`.
  Loading a SavedModel that contains it may require
  `custom_objects={'Attention': Attention}`.

`nltk.TweetTokenizer` needs the NLTK data that ships with the tokenizer
(usually present after `pip install nltk`). No extra `nltk.download`
corpus is required for that class.

### Path mismatches inside the notebooks

| Notebook | Sentence path it uses | What the repo actually has |
| --- | --- | --- |
| `baseline_models.ipynb` | `train_sentence.csv` in CWD | `dataset/train_sentence.csv` |
| `evaluate_loaded_dl_models.ipynb` | same bare filenames | same |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` | correct |

If you revive the first two notebooks, either `cd dataset` (and fix the
model paths) or prefix `dataset/`.

`baseline_models.ipynb` also writes pickles under `baseline_models/`,
which is currently an empty directory.

### Suggested revival order

1. Install [`requirements.txt`](../requirements.txt) in a fresh virtualenv.
2. Download GloVe Twitter 200-d and convert it if you only have the text
   file (`gensim.scripts.glove2word2vec` or a short Python loop).
3. Run `ReadOpen` + `Preprocess` on `dataset/train_*.csv` and confirm the
   padded width is 78 if you want to compare against the saved summaries.
4. Rebuild `PrepModel` and train; do not expect the 2023 weights to load
   until the SavedModel directories are complete.
5. Evaluate with the same metrics as the notebook: accuracy, F1,
   precision, recall on test and subtest, sarcastic class = `1`.

### Determinism

`ml_read_data` shuffles with an unseeded `numpy.random.permutation`.
Classical training order will move. The neural preprocessor does not
shuffle. sklearn models may still differ across versions even with a
seed.

## What "done" meant for the course

The graded artifact was the notebook trail plus the two BiLSTM
checkpoints and the metric tables. It was not a packaged library. The
example package in this branch is documentation support, not a rewrite of
that training job.
