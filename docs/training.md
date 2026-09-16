# Training

This page is the 2023 training recipe as it actually exists in the notebooks, not a cleaned-up rewrite. The code does not live in a `train.py`; it lives in `baseline_models.ipynb` (sklearn) and in whatever cells originally called `PrepModel.fit` (the fit cells were not checked in; only the saved Keras directories remain).

## What you need on disk

| Artifact | Required to train | In this repo? |
| --- | --- | --- |
| `dataset/train_*.csv` | Yes | Yes |
| `dataset/test_*.csv` | Used as the validation handle in the DL notebook | Yes |
| GloVe Twitter 200-d (`glove.twitter.27B.200d.bin` or `glove_tt.txt`) | Yes | **No** |
| `emoji2vec_twitter.bin` | Yes for any `*_we` / multi-modal run | Yes |
| `nltk` + `punkt` / tweet tokenizer models | Yes (`ReadOpen`) | Install locally |
| `emoji` package | Yes (`Preprocess` emoji fallback) | Install locally |
| TensorFlow 2.x + `keras_preprocessing` | Bi-LSTM only | Install locally |
| scikit-learn + joblib | Baselines only | Install locally |

See `requirements.txt` at the repo root for a pinned-enough list. The original run was Keras 2 / TF 2 on a 2023 laptop-class GPU; the attention docstring still mentions Keras 2.0.6 as the layer’s original test target.

## Classical models

`baseline_models.ipynb` does this, twice per algorithm (single-modal then multi-modal):

```text
glove  = KeyedVectors.load_word2vec_format('glove.twitter.27B.200d.bin', binary=True)
emoji  = KeyedVectors.load_word2vec_format('emoji2vec_twitter.bin', binary=True)
X, y, X_emoji, y_emoji = ml_read_data('train_sentence.csv', 'train_label.csv', glove, emoji)
clf.fit(X, y)            # 200-d
clf_we.fit(X_emoji, y_emoji)  # 400-d
joblib.dump(...)
```

Details that are easy to miss:

1. **Paths.** The baseline notebook reads `train_sentence.csv` from the *current working directory*. `get_metrics_of_models.ipynb` reads `dataset/train_sentence.csv`. Run notebooks from the repo root or fix the paths.
2. **Shuffle.** `ml_read_data` permutes rows with an unseeded `numpy.random.permutation`. Two fits are not comparable unless you save the permutation or the pickle.
3. **No hyperparameter search.** `SVC()`, `DecisionTreeClassifier()`, `RandomForestClassifier()`, `GradientBoostingClassifier()` are called with sklearn defaults.
4. **Load-or-fit.** Each cell is wrapped in `try: joblib.load(...) except FileNotFoundError: fit; dump`. If a pickle exists, the cell will **not** retrain even if you changed the features.
5. **One notebook bug.** The Decision Tree *multi-modal* branch in `baseline_models.ipynb` constructs `SVC()` instead of `DecisionTreeClassifier()`. The checked-in `dt_classifier_we.pkl` may therefore be an SVM in a DT-shaped filename. Treat the DT multi-modal row in [evaluation.md](evaluation.md) with that caveat; the metrics notebook loaded whatever pickle was on disk.

Suggested command once GloVe is present (not run in this docs-only change):

```python
from sklearn.ensemble import RandomForestClassifier
from gensim.models import KeyedVectors
from data_utils import ml_read_data
import joblib

glove = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
emoji = KeyedVectors.load_word2vec_format("emoji2vec_twitter.bin", binary=True)
X, y, X_e, y_e = ml_read_data(
    "dataset/train_sentence.csv", "dataset/train_label.csv", glove, emoji
)
rf = RandomForestClassifier(n_jobs=-1, random_state=0)
rf.fit(X, y)
joblib.dump(rf, "baseline_models/rf_classifier.pkl")
```

Adding `random_state` is the one change worth making if you retrain.

## Bi-LSTM + Attention

`dl_model.PrepModel` only *builds* the graph. A training loop that matches the saved models looks like this:

```python
from data_utils import ReadOpen, Preprocess
from dl_model import PrepModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

docs, labels, count = ReadOpen("dataset/train_sentence.csv", "dataset/train_label.csv")
padded, matrix, maxlen, tokenizer = Preprocess(
    docs, count, glove_model, emoji2vec_model, get_emoji2vec=True
)
model = PrepModel(count, matrix, maxlen, lrate=0.001)
ckpt = ModelCheckpoint(
    "model/best_model_multi_modal",
    monitor="val_acc",
    save_best_only=True,
)
model.fit(
    padded,
    labels,
    validation_data=(padded_test, labels_test),
    epochs=8,            # original notebook range; stop early
    batch_size=64,
    callbacks=[ckpt, EarlyStopping(monitor="val_acc", patience=2)],
)
```

The evaluation notebook assigns `X_val = X_test`. There is no third split. Early stopping on test is a course-project shortcut; do not treat the 0.8735 number as a nested-CV estimate.

### Single-modal twin

Call `Preprocess(..., get_emoji2vec=False)` so emoji / OOV rows stay zero, then `PrepModel` and the same fit. Everything else — dropout, LSTM width, Adam `lr=0.001`, binary cross-entropy — stays identical. That is why the two saved models have the same trainable size.

### Learning-rate argument

`PrepModel` uses `Adam(lr=lrate)`. Recent TensorFlow prefers `learning_rate=`. If you rebuild on TF 2.14+, change that one keyword or the constructor will error.

### Attention length lock

Do not change `maxlen` between train and the checkpoint you want to reload. `Attention` allocates a bias of shape `(timesteps,)`. The 2023 checkpoints expect **78** steps (see the `sequential_5` / `sequential_6` summaries).

## Class imbalance

Train is 53.5% / 46.5%. The original fit does not pass `class_weight`. For a retrain, `{0: 1.0, 1: 21292/18488}` is the honest baseline. It will move recall more than accuracy.

## What this docs change does *not* do

- It does not retrain the Bi-LSTMs (GloVe is not in the repo; a full fit is outside the scope of a documentation pass).
- It does not rewrite `data_utils.py` / `dl_model.py` to add seeds.
- It does not download GloVe.

Use `examples/cue_baseline.py` if you want a number you can reproduce from the CSVs alone. Use `examples/reprint_course_results.py` to print the 2023 table without loading any model.
