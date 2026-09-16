# Reproduction notes

This archive is a **2023 course snapshot**, not a pinned research release. You can reproduce the *ideas* with the example suite today. You can reproduce the *tables* only if you restore a few files that never landed in git.

## What is complete

- Sentence / label CSVs for train, test, subtest
- `data_utils.py`, `dl_model.py`, `attention_layer.py`
- Executed notebooks with metric stdout
- `emoji2vec.bin`, `emoji2vec_twitter.bin`
- Decision Tree and Gradient Boosting pickles (word and word+emoji)
- Example scripts + unit tests (stdlib + NumPy)

## What is missing

| Artifact | Referenced as | Effect |
| --- | --- | --- |
| GloVe Twitter 27B 200-d | `glove.twitter.27B.200d.bin` or `glove_tt.txt` | Cannot rebuild embedding matrices |
| SVM pickles | `baseline_models/svm_model.pkl`, `svm_model_we.pkl`, `svm_classifier*.pkl` | Notebooks fall back to `SVC().fit` if you have features |
| Random Forest pickles | `baseline_models/rf_classifier.pkl`, `rf_classifier_we.pkl` | Same |
| Keras variable shards | `model/*/variables/variables.data-*` | `load_model` cannot restore weights |
| Training loop | no `fit()` in `dl_model.py` | Epochs / batch size / callbacks unknown |
| Seeds | none | `ml_read_data` shuffles with unseeded `np.random` |

## Restore GloVe

1. Download `glove.twitter.27B.200d.txt` from the [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/).
2. Convert to word2vec binary if you want the path the first notebooks use:

```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove2word2vec("glove.twitter.27B.200d.txt", "glove.twitter.27B.200d.w2v.txt")
kv = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.w2v.txt", binary=False)
kv.save_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
```

3. Or skip conversion and point `get_metrics_of_models.ipynb` at a text file, as that notebook already did with `glove_tt.txt`.

Gensim 4 renamed `model.vocab` to `model.key_to_index` and `model[word]` still works, but `if j in model_word2vec.vocab` in `data_utils.py` will raise. Patch:

```python
vocab = getattr(model, "key_to_index", None) or model.vocab
if j in vocab:
    ...
```

## Restore a trainable deep model

```python
from data_utils import ReadOpen, Preprocess, preprocess_test
from dl_model import PrepModel

docs, y, count = ReadOpen("dataset/train_sentence.csv", "dataset/train_label.csv")
X, emb, maxlen, tok = Preprocess(docs, count, glove, emoji2vec, get_emoji2vec=True)
model = PrepModel(count, emb, maxlen, lrate=0.001)
# model.fit(X, y, batch_size=32, epochs=?, validation_split=0.1)
```

Modern TensorFlow wants `Adam(learning_rate=0.001)` and `tensorflow.keras.layers.Embedding`. The dual import of `Embedding` in `dl_model.py` (one unused) is leftover.

Because `Attention.b` is sized to `maxlen`, export the tokenizer and `maxlen` next to any new checkpoint.

## Reload the shipped sklearn trees

```python
import joblib
dt = joblib.load("baseline_models/dt_classifier.pkl")
dt_we = joblib.load("baseline_models/dt_classifier_we.pkl")
```

Those pickles expect the **same 200-d / 400-d mean-pooled features** `ml_read_data` produced. Feature dim mismatch will error. They also pin an old sklearn pickle protocol; use sklearn 1.2.x first if 1.5+ refuses to load.

## Example-only reproduction (this PR)

No GloVe, no TF:

```bash
python3 -m pip install -r requirements-examples.txt
python3 -m unittest discover -s tests -v
python3 examples/inspect_dataset.py --root dataset --out docs/generated/dataset_report.md
python3 examples/lexical_baseline.py --root dataset --out docs/generated/lexical_baseline.md
```

These commands are the regression surface for the documentation. They must stay green on a stock CPython + NumPy box.

## Known code smells to leave or fix later

Personal-project debt, documented so a future edit does not “discover” it:

- `baseline_models.ipynb` trains the *wrong estimator* in several `except FileNotFoundError` branches (DT multi-modal fits `SVC()`, RF single-modal fits `SVC()`, GBT single-modal fits `SVC()`). The successful 2023 run loaded pickles and never hit those branches.
- `AverageVectorPerTweet` and `AverageVectorPerEmoji` are copy-paste twins.
- Bare `except:` in `Preprocess` swallows emoji2vec `KeyError`s.
- `PrepModel` uses `Adam(lr=...)`.
- SavedModel directories are graphs without weights.

None of that is changed in the docs/examples pass on purpose: this PR explains the archive, it does not silently rewrite the 2023 experiment.
