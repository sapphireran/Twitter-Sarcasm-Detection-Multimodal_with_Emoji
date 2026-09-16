# Reproducing the 2023 notebooks

This page is for the original Keras / sklearn pipeline. If you only want the docs examples, skip to the last section.

## What the checkout already contains

Present:

- All six dataset files
- `emoji2vec.bin` and `emoji2vec_twitter.bin`
- Keras SavedModel folders `model/best_model_single_modal` and `model/best_model_multi_modal` (graphs + `keras_metadata.pb` only; **no `variables/` shards** in this checkout)
- Decision-tree and gradient-boosting pickles under `baseline_models/`

Missing from this checkout:

- `glove.twitter.27B.200d.bin` (expected by `baseline_models.ipynb` and `evaluate_loaded_dl_models.ipynb`)
- `glove_tt.txt` (expected by `get_metrics_of_models.ipynb`)
- `baseline_models/svm_classifier.pkl`, `svm_classifier_we.pkl`
- `baseline_models/rf_classifier.pkl`, `rf_classifier_we.pkl`
- and the `svm_model*.pkl` / `svm_model_we.pkl` names used in the metrics notebook

You can evaluate the saved Keras models only after you have a GloVe file and a matching preprocess run, because the checkpoints consume **integer padded sequences**, not raw text.

## Environment

`requirements.txt` lists the course-project stack. Two API landmines:

1. **Gensim 4** removed `KeyedVectors.vocab`. `data_utils.py` still writes `if j in model_word2vec.vocab`. Use Gensim 3.8.x or change those tests to `j in model_word2vec`.
2. **`keras_preprocessing`** is a separate package from `tensorflow.keras.preprocessing`. The import in `data_utils.py` is the standalone package.
3. **`Adam(lr=lrate)`** is the TF 2.8-era keyword. Recent Keras wants `learning_rate=`.
4. **`tensorflow.python.keras.layers.embeddings.Embedding`** (imported in `dl_model.py` and then unused) disappeared from public TF 2.16+ layouts.

A practical working window is TensorFlow 2.8–2.15, Gensim 3.8, Python 3.8–3.10. This Cloud Agent image is Python 3.12 without TensorFlow; do not treat a failed local import here as a broken checkpoint.

## Data paths

Notebooks disagree about the working directory:

- `baseline_models.ipynb` and `evaluate_loaded_dl_models.ipynb` open `train_sentence.csv` in the **current directory**
- `get_metrics_of_models.ipynb` opens `dataset/train_sentence.csv`

Copy or symlink the six files next to the notebook you are running, or cd into `dataset/` and point the model paths back up one level.

## GloVe

The official Stanford release is `glove.twitter.27B.200d.txt` (plain text). The notebooks call `load_word2vec_format(..., binary=True)` on a `.bin` file, so someone converted it to word2vec binary during the course. `get_metrics_of_models.ipynb` instead loads `glove_tt.txt` with `binary=False`.

Either format is fine if the vectors are 200-d and the keys match the lowercased `TweetTokenizer` output, including tokens such as `<user>` if you want mention rows to be nonzero.

## Evaluating a saved Keras model

```python
from gensim.models import KeyedVectors
from data_utils import ReadOpen, Preprocess, preprocess_test
import tensorflow as tf

glove = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
emoji = KeyedVectors.load_word2vec_format("emoji2vec_twitter.bin", binary=True)

docs, y, n = ReadOpen("dataset/train_sentence.csv", "dataset/train_label.csv")
padded, matrix, maxlen, tok = Preprocess(docs, n, glove, emoji, get_emoji2vec=True)

test_docs, y_test, _ = ReadOpen("dataset/test_sentence.csv", "dataset/test_label.csv")
x_test = preprocess_test(tok, maxlen, test_docs)

model = tf.keras.models.load_model(
    "model/best_model_multi_modal",
    custom_objects={"Attention": __import__("attention_layer").Attention},
)
model.evaluate(x_test, y_test)
```

You may need `custom_objects` depending on how the SavedModel was exported. The recorded `evaluate_loaded_dl_models.ipynb` call is `tf.keras.models.load_model("model/best_model_multi_modal")` with no extra objects, which worked on that machine.

## Retraining the deep model

```python
from dl_model import PrepModel

model = PrepModel(count=n, embedding_matrix=matrix, l=maxlen, lrate=0.001)
model.fit(padded, y, epochs=8, batch_size=64, validation_split=0.1)
```

The original epoch / batch-size choices are not stored in a script. The numbers above are a reasonable starting point, not a claim about the saved checkpoints.

## Retraining sklearn baselines

```python
from data_utils import ml_read_data
from sklearn.ensemble import RandomForestClassifier

X, y, X_e, y_e = ml_read_data(
    "dataset/train_sentence.csv",
    "dataset/train_label.csv",
    glove,
    emoji,
)
rf = RandomForestClassifier()
rf.fit(X, y)
```

Set `numpy.random.seed(...)` before `ml_read_data` if you want a repeatable shuffle. Fix the constructor mixups in `baseline_models.ipynb` before using it as a training script (see [methodology.md](methodology.md)).

## NLTK data

`TweetTokenizer` lives in the `nltk` package. The first call may require `nltk.download("punkt")` on some versions; `TweetTokenizer` itself does not need the punkt model.

## Docs examples (this environment)

No GloVe, no TensorFlow, no NLTK:

```bash
python3 examples/inspect_dataset.py
python3 examples/emoji_signals.py --top 15
python3 examples/lexical_baseline.py
python3 examples/lexical_baseline.py --strip-supervision-tags
python3 examples/attention_demo.py
python3 examples/preprocess_walkthrough.py
python3 -m unittest discover -s tests -v
```

Dependencies: Python 3.10+ and NumPy (`requirements-examples.txt`).
