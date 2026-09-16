# Reproduction (2023 notebooks)

This is the original course path. For the stdlib walkthrough, use
[examples.md](examples.md) instead.

## What you already have

* `dataset/*.csv` — train / test / subtest
* `emoji2vec.bin`, `emoji2vec_twitter.bin`
* `model/best_model_{single,multi}_modal` Keras SavedModels
* `baseline_models/dt_classifier*.pkl`, `gbt_classifier*.pkl`
* Notebooks: `baseline_models.ipynb`, `evaluate_loaded_dl_models.ipynb`,
  `get_metrics_of_models.ipynb`

## What you still need

1. Python 3.8–3.10 is the safest match for the TF 2.8-era notebooks.
   The attention layer uses `tf.keras` internals (`backend as K`,
   `input_length` on `Embedding`, `Adam(lr=...)`) that newer Keras 3
   will reject without edits.
2. `pip install -r requirements.txt` (TensorFlow, gensim, nltk, emoji,
   pandas, scikit-learn, joblib, keras-preprocessing).
3. `nltk.download('punkt')` if TweetTokenizer asks for it (often it
   does not).
4. Stanford **GloVe Twitter 27B 200d**. Convert to word2vec binary and
   save as `glove.twitter.27B.200d.bin` in the repo root (as
   `baseline_models.ipynb` does) *or* as `glove_tt.txt` (as
   `get_metrics_of_models.ipynb` does). That file is too large for git.

## Evaluate the saved BiLSTMs

`evaluate_loaded_dl_models.ipynb`:

1. Load GloVe + emoji2vec with gensim `KeyedVectors`
2. `ReadOpen` + `Preprocess` on train to rebuild the tokenizer / matrix
3. `preprocess_test` on test and subtest
4. `tf.keras.models.load_model("model/best_model_multi_modal")` and
   `.../best_model_single_modal`
5. `model.evaluate(X_test, y_test)` / `... subtest`

Expect test acc ≈ 0.8735 (multi) and 0.8635 (single). Custom objects:
if load fails on `Attention`, pass
`custom_objects={"Attention": Attention}` from `attention_layer.py`.

## Retrain classical models

`baseline_models.ipynb` tries to `joblib.load` from `baseline_models/`
and trains on cache miss. SVM and random-forest pickles are missing in
this checkout, so those cells will train (slow on 39k × 200-d SVM).
Decision tree and GBT pickles are present.

`ml_read_data` shuffles every call. Evaluate in the same session.

## Retrain the BiLSTM

There is no dedicated train notebook in the root — `dl_model.PrepModel`
is the constructor. A minimal loop:

```python
from data_utils import ReadOpen, Preprocess, preprocess_test
from dl_model import PrepModel

docs, y, count = ReadOpen("dataset/train_sentence.csv", "dataset/train_label.csv")
padded, matrix, maxlen, tokenizer = Preprocess(docs, count, glove, emoji2vec)
model = PrepModel(count, matrix, maxlen, lrate=0.001)
model.fit(padded, y, epochs=5, batch_size=32, validation_split=0.05)
```

Use `get_emoji2vec=False` in `Preprocess` for the single-modal embedding
table. This is GPU-optional but slow on CPU.

## Known sharp edges

* `from tensorflow.python.keras.layers.embeddings import Embedding`
  in `dl_model.py` is the TF 2.x private path. Prefer
  `tensorflow.keras.layers.Embedding` on a clean install.
* `Adam(lr=lrate)` is deprecated; Keras 3 wants `learning_rate=`.
* `model_word2vec.vocab` in `data_utils.py` is Gensim 3. Gensim 4 uses
  `key_to_index`. Pin gensim `<4` or patch those lookups.
* Sentence CSVs are not RFC-4180: `ReadOpen` does not use a CSV parser.

## What not to do

Do not point these notebooks at live Twitter/X APIs. The project is
offline: local CSVs plus local embedding files.
