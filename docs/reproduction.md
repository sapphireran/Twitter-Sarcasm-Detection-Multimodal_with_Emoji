# Reproduction notes

This page is the honest setup guide for the 2023 notebooks and for the
newer personal examples.

## What you can run from this clone

These need only the Python standard library plus NumPy (already used by
the original project):

```bash
python3 -m examples.inspect_dataset
python3 -m examples.reported_results
python3 -m examples.attention_walkthrough
python3 -m examples.multimodal_fusion
python3 -m examples.embedding_matrix_walkthrough
python3 -m examples.tokenize_tweets --split subtest --limit 5
python3 -m examples.lexical_baseline
python3 -m unittest discover -s tests -v
```

They read `dataset/*.csv` and the in-repo results catalog. They do not
download GloVe, start TensorFlow, or load the pickled sklearn models.

## What the 2023 notebooks expected

```text
Python 3.x of that year
tensorflow / keras  (TF 2.x with tensorflow.python.keras shims)
keras_preprocessing
gensim KeyedVectors
nltk TweetTokenizer
emoji
scikit-learn, joblib
pandas, numpy
```

`requirements.txt` at the repository root lists that stack. Versions are
lower bounds that match the import surface, not a lockfile from 2023.

### External files that are not in git

| File | Used by | Notes |
| --- | --- | --- |
| `glove.twitter.27B.200d.bin` | both notebooks | Binary KeyedVectors export of GloVe Twitter 27B, 200-d |
| `glove_tt.txt` | `get_metrics_of_models.ipynb` | Text GloVe table, same geometry |
| `emoji2vec_twitter.bin` | notebooks | **Is** in the repo (≈1.3 MB) |
| `emoji2vec.bin` | unused by the notebooks | Upstream emoji2vec release, also in the repo |
| `baseline_models/*.pkl` | notebooks | Decision tree and GBT pickles are present; SVM / RF pickles are not |
| `model/best_model_*/variables/` | `tf.keras.models.load_model` | Missing. Only the protocol-buffer graph files were uploaded |

Without GloVe you cannot rebuild `X` / `X_emoji` or the Keras embedding
matrix. Without the variable shards you cannot reload the 2023 Bi-LSTMs.

### Path inconsistency

`baseline_models.ipynb` and `evaluate_loaded_dl_models.ipynb` open
`train_sentence.csv` from the **current working directory**.
`get_metrics_of_models.ipynb` opens `dataset/train_sentence.csv`. If you
re-run a notebook, `cd` into `dataset/` for the first two or edit the
paths.

### Gensim API

`data_utils.py` uses `model.vocab` and `model[token]`. Gensim 4 moved
vocabulary membership to `key_in_wv` / `"token" in kv`. Either pin
gensim `<4` or change the lookups before retraining.

### Keras optimizer argument

`Adam(lr=0.001)` needs `learning_rate=0.001` on current TensorFlow.

## Suggested full retrain (original stack)

1. Install `requirements.txt` in a fresh virtualenv.
2. Download [GloVe Twitter 27B](https://nlp.stanford.edu/projects/glove/)
   and convert the 200-d vectors to word2vec binary, **or** point
   `KeyedVectors.load_word2vec_format(..., binary=False)` at the `.txt`.
3. `nltk.download("punkt")` is not required; `TweetTokenizer` ships with
   NLTK. Still run `import nltk; nltk.download('punkt')` only if you add
   other tokenizers.
4. From the repo root, with data paths aimed at `dataset/`:
   - rebuild classical features via `ml_read_data`
   - `PrepModel(...)` then `model.fit` with a shuffled train set and the
     official test set as validation (the 2023 notebooks used `X_val = X_test`)
5. Evaluate on `test_*` and on the emoji subset (`subtest_*`).
6. Compare against the frozen table in [experiments.md](experiments.md).
   Expect drift: sklearn and TF defaults have moved.

Using the official test set as validation is a course-project shortcut.
Do not treat those checkpoints as an unbiased model-selection result.

## Suggested lightweight retrain (personal examples)

If you only want a number on this machine:

```bash
python3 -m examples.lexical_baseline --max-train 8000
```

That script standardizes surface features and fits a logistic classifier
implemented in NumPy. It is for pedagogy, not for beating the 2023 LSTM.

## Randomness

`ml_read_data` calls `np.random.permutation` with whatever global NumPy
state is current. The notebooks do not seed it. Classical metrics in the
repo are from one such shuffle plus sklearn defaults of that year. The
neural metrics come from saved models scored after a separate preprocess
that does **not** shuffle evaluation order, so those accuracy numbers are
deterministic given the weights and the tokenizer.

The example lexical baseline seeds NumPy (`default 7`) so repeat runs
match.
