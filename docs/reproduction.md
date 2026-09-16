# Reproduction notes

The 2023 notebooks are the source of the published numbers. This
page lists what you need, in what order, and what will fail if you
clone the repo as it exists today.

## What is on disk vs what the notebooks expect

| Artifact | In git? | Notes |
| --- | --- | --- |
| `dataset/*.csv` | yes | 39,780 / 2,000 / 278 rows |
| `emoji2vec_twitter.bin` | yes | gensim `KeyedVectors` binary |
| `emoji2vec.bin` | yes | unused by the notebooks |
| `glove.twitter.27B.200d.bin` | **no** | convert GloVe Twitter 27B 200-d yourself |
| `glove_tt.txt` | **no** | text GloVe used by one metrics notebook cell |
| `baseline_models/dt_*.pkl`, `gbt_*.pkl` | yes | sklearn pickles |
| `baseline_models/svm_*.pkl`, `rf_*.pkl` | **no** | notebooks will retrain or crash |
| `model/best_model_*/saved_model.pb` | partial | graph only; `variables/` is missing |
| `nltk` tweet tokenizer | **no** | `nltk.download` needed |

## Environment

`requirements.txt` lists the stack the notebooks import. The original
run used Keras 2-style APIs:

- `Adam(lr=0.001)` — modern Keras wants `learning_rate`.
- `from tensorflow.python.keras.layers.embeddings import Embedding`
  and `tensorflow.python.keras.layers.core` — those internal paths
  moved or vanished after TF 2.15.
- `gensim` `KeyedVectors.vocab` — removed in gensim 4. Use
  `word in kv` / `kv.key_to_index` instead, or pin gensim 3.8.x.
- `keras_preprocessing.text.Tokenizer` — still available as a
  standalone package.

Practical pin if you want the least friction: Python 3.8–3.10,
`tensorflow==2.10`, `gensim==3.8.3`, `nltk==3.7`.

## Notebook order

1. Obtain GloVe Twitter 27B 200-d and convert it to word2vec binary
   (`gensim.scripts.glove2word2vec` then `KeyedVectors.save_word2vec_format`).
2. `nltk.download('punkt')` is not enough; `TweetTokenizer` ships
   with nltk but you still need the `nltk` package installed.
3. `baseline_models.ipynb` — loads or trains the four sklearn pairs.
   Paths in that notebook are **bare filenames** (`train_sentence.csv`),
   so either `cd dataset` or edit the paths to `dataset/...`.
4. `evaluate_loaded_dl_models.ipynb` — rebuilds the Keras sequences
   and calls `load_model` on `model/best_model_{single,multi}_modal`.
   Will fail without `variables/variables.data-*` plus
   `variables/variables.index`.
5. `get_metrics_of_models.ipynb` — same embeddings, then prints the
   tables copied into [results.md](results.md). One cell loads
   `glove_tt.txt` (text) instead of the `.bin` used elsewhere.

## Training the deep model from scratch

`dl_model.PrepModel` is the graph. There is no dedicated training
notebook in the tree; the 2023 fit lived outside these files. A
minimal loop that matches the compiled model:

```python
from dl_model import PrepModel
from data_utils import ReadOpen, Preprocess, preprocess_test

# after loading GloVe + emoji2vec and calling Preprocess on train:
model = PrepModel(count_train, embedding_matrix, maxlen, lrate=0.001)
model.fit(
    padded_train, labels_train,
    validation_data=(padded_test, labels_test),
    epochs=8,
    batch_size=64,
)
```

`count` passed into `PrepModel` / `Embedding` must be the **first
dimension of `embedding_matrix`**, which `Preprocess` sizes as
`zeros((count, 200))` where `count` is the number of **training
lines**, not `len(tokenizer.word_index) + 1`. That is an off-by-
construction choice from the coursework: the matrix is wider than
the true vocab and the unused tail stays zeros. Do not “fix” it to
`vocab+1` without also changing `Preprocess`.

## Known quirks (leave them; document them)

1. **Fallback classifiers are wrong** in `baseline_models.ipynb`.
   RF/GBT single-modal and DT multi-modal `except` branches
   instantiate `SVC()`. Only trust numbers from the 2023 pickles
   or from a corrected retrain.
2. **Unseeded shuffle** in `ml_read_data`. Two runs write different
   row orders into `X` / `y`. Metrics should be similar; exact
   floats will not match.
3. **Comma stripping** in `ReadOpen` changes tokenization for any
   tweet that used commas as CSV quotes.
4. **Bare `except:`** in `Preprocess` swallows missing emoji2vec
   keys and writes zeros.
5. **SavedModel shards missing.** The `*.pb` files are not enough
   to evaluate. Retrain if you need a loadable checkpoint.
6. **Chinese comments** in `get_metrics_of_models.ipynb` are from
   the original write-up session; they do not change behavior.

## What you can reproduce without GloVe

From the repo root, with only NumPy:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/inspect_dataset.py
python3 examples/tokenize_tweets.py
python3 examples/toy_embedding_fusion.py
python3 examples/attention_walkthrough.py
python3 examples/lexical_sarcasm_baseline.py
python3 -m unittest discover -s tests -v
```

Those scripts re-implement the *ideas* (mean-pool, concat, Raffel
attention, lexical floor) on the real CSV files. They do not claim
to match the 2023 floats.
