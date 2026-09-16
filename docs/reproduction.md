# Reproduction

This repository is a 2023 course dump, not a packaged trainer. The
notebooks assume a local Keras 2 / TensorFlow 2 environment, Gensim 3
keyed vectors, and a GloVe Twitter binary that was never committed.

If you only want to understand the pipeline, skip the heavy stack and
run the examples (NumPy + the CSVs in this repo):

```bash
python -m unittest discover -s tests -v
python examples/run_all.py
```

## Repository layout the notebooks expect

Some cells use paths relative to `dataset/`:

```python
ml_read_data("dataset/train_sentence.csv", "dataset/train_label.csv", ...)
```

Others assume you copied the CSVs into the working directory:

```python
ml_read_data("train_sentence.csv", "train_label.csv", ...)
```

The embedding binaries are always loaded from the working directory:

```python
KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
KeyedVectors.load_word2vec_format("emoji2vec_twitter.bin", binary=True)
```

`emoji2vec_twitter.bin` ships with the repo. GloVe does not.

## Fetch GloVe Twitter 200-d

The official release is `glove.twitter.27B.zip` from the
[GloVe project](https://nlp.stanford.edu/projects/glove/). The 200-d
file inside the zip is text, not word2vec-binary. Two ways the 2023
notebooks consumed it:

1. Convert with Gensim and save `glove.twitter.27B.200d.bin`, then load
   with `binary=True` (baseline + eval notebooks).
2. Load the text file directly as `glove_tt.txt` with `binary=False`
   (`get_metrics_of_models.ipynb`).

Sketch of the conversion, once you have the `.txt` file:

```python
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec("glove.twitter.27B.200d.txt", "glove.twitter.27B.200d.w2v.txt")
model = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.w2v.txt")
model.save_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
```

Do not commit the resulting binary; it is far larger than GitHub's
file limit.

## Python dependencies (2023 stack)

[`requirements.txt`](../requirements.txt) lists the libraries the
original files import. Versions that are known to matter:

| Library | Why the pin is fussy |
| --- | --- |
| TensorFlow / Keras 2 | `attention_layer.py` uses `tf.keras.backend` and a custom `Layer` |
| `keras_preprocessing` | `pad_sequences` / `Tokenizer` in `data_utils.py` |
| Gensim 3.x | `if token in model.vocab` — crashes on Gensim 4 |
| `emoji` | `emoji.emoji_list` / `emoji.is_emoji` in `Preprocess` |
| NLTK | `TweetTokenizer`; run `nltk.download("punkt")` only if you stray off tweets |
| scikit-learn + joblib | baseline pickles |

On Gensim 4, change every `.vocab` membership test in `data_utils.py`
to `token in model` before running `ml_read_data`.

On Keras 3 / recent TensorFlow, `PrepModel` needs `Adam(learning_rate=)`
instead of `Adam(lr=)`, and the custom `Attention` layer needs a
`get_config` if you want `load_model` to round-trip it.

## What you can run today without GloVe

| Command | Needs |
| --- | --- |
| `python examples/dataset_overview.py` | stdlib + NumPy |
| `python examples/tokenize_demo.py` | stdlib |
| `python examples/embedding_demo.py` | NumPy |
| `python examples/attention_demo.py` | NumPy |
| `python examples/sarcasm_cues.py` | stdlib |
| `python examples/toy_pipeline.py` | NumPy |
| `python -m unittest discover -s tests -v` | NumPy |

## What you cannot run from this snapshot alone

| Target | Blocker |
| --- | --- |
| Retrain sklearn baselines to match the notebook | missing GloVe; `ml_read_data` is unseeded |
| Load `model/best_model_*` | SavedModel `variables/` weights were not uploaded |
| Load SVM / Random Forest pickles | files are referenced but not in `baseline_models/` |
| Exact Gensim 3 `model.vocab` checks | current Gensim is 4.x in most installs |

Decision Tree and Gradient Boosting pickles *are* in
`baseline_models/`, but they still expect 200-d / 400-d inputs built
from GloVe + emoji2vec, so they are not useful without those tables.

## Suggested order if you do rebuild

1. Install the 2023-ish stack from `requirements.txt` into a dedicated
   virtualenv.
2. Download GloVe Twitter 200-d and convert it as above.
3. Confirm `emoji2vec_twitter.bin` loads:
   `KeyedVectors.load_word2vec_format("emoji2vec_twitter.bin", binary=True)`.
4. Patch `data_utils.py` for your Gensim major version.
5. Run `examples/dataset_overview.py` and check the row counts against
   [dataset.md](dataset.md) so you know you are on the same CSVs.
6. Only then open the notebooks.

The example toy pipeline is the supported way to see single-modal vs
multi-modal behavior in this checkout.
