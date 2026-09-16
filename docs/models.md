# Models

Five classifiers appear in the notebooks. Four are sklearn baselines on 200-d / 400-d averages. One is a frozen-embedding Bi-LSTM with attention on padded token ids.

## Shared training facts

- Labels are binary. Every model emits a class, not a calibrated probability (except the LSTM sigmoid, which is only used as `acc` / threshold 0.5 in `evaluate`).
- Classical models are saved with `joblib` under `baseline_models/`. The upload is incomplete: SVM and RF pickles that `get_metrics_of_models.ipynb` loads (`svm_model.pkl`, `rf_model.pkl`) are not in git. DT and GBT pickles are.
- Deep models are SavedModel directories under `model/`.
- No class weights. Train is close enough to balanced (46.5% sarcastic) that the notebooks never set `class_weight`.

## SVM

```python
SVC()   # sklearn defaults: RBF, C=1, gamma='scale'
```

Two instances: `svm_classifier` on 200-d, `svm_classifier_we` on 400-d (`we` = "with emoji").

Headline: 0.769 / 0.763 test accuracy. The extra 200 dimensions do not help SVM on the main test set. They help a little on the subtest (0.813 → 0.824). RBF SVM in 400-d with 40 k rows is also the slowest baseline to fit, which is why the notebook prefers a cached pickle.

## Decision tree

```python
DecisionTreeClassifier()   # defaults: Gini, grow until pure
```

Unpruned trees overfit 200-d averages. Test accuracy 0.727 / 0.730 is the weakest of the five. Recall is high (0.85) and precision is low (0.68): the tree labels too many tweets sarcastic.

**Bug in the fallback path.** `baseline_models.ipynb` trains `SVC()` for the multimodal branch if the pickle is missing. See [known-issues.md](known-issues.md). The executed cells in that notebook loaded pickles, so the reported DT numbers are still DT vs DT.

## Random forest

Best classical model.

```python
RandomForestClassifier()   # defaults: 100 trees, Gini, bootstrap
```

Test 0.815 / 0.818. Subtest 0.806 / 0.853. The +4.7 subtest accuracy lift is the largest classical multimodal delta in the notebooks.

**Same fallback bug:** the single-modal `except FileNotFoundError` branch fits an `SVC()`, not a forest. Cached pickles again saved the reported run.

## Gradient boosting

```python
GradientBoostingClassifier()   # defaults: 100 trees, lr=0.1
```

Test 0.746 / 0.748. Subtest is flat at 0.795 either way. F1 on the subtest even dips slightly when emoji are added (0.838 → 0.836). GBT is the baseline that does **not** support the multimodal claim.

Fallback bug: single-modal branch fits `SVC()`.

## Bi-LSTM + attention

Defined in `dl_model.py` as `PrepModel(count, embedding_matrix, l, lrate=0.001)`.

| Layer | Output | Trainable? |
| --- | --- | --- |
| `Embedding(count, 200, weights=E, input_length=l)` | `(B, 78, 200)` | no |
| `Dropout(0.25)` | same | — |
| `Bidirectional(LSTM(256, return_sequences=True))` | `(B, 78, 512)` | yes (935,936) |
| `Dropout(0.4)` | same | — |
| `Bidirectional(LSTM(256, return_sequences=True))` | `(B, 78, 512)` | yes (1,574,912) |
| `Dropout(0.4)` | same | — |
| `Attention()` | `(B, 512)` | yes (512 + 78 if bias) |
| `Dense(1, sigmoid)` | `(B, 1)` | yes (513) |

`summary()` on the saved graphs reports **2,510,848** total parameters, all marked trainable. That disagrees with `trainable=False` on the embedding in `PrepModel`. SavedModel wrapping (`module_wrapper_*`) plus a TF version skew is the likely cause: the embedding weights exist, but the freeze flag may not have survived export. Either way the two saved graphs have the same size; they differ in the *values* of \(E\), not in architecture.

LSTM options copied from the 2023 Keras 2 defaults the author pinned:

- `kernel_initializer='he_normal'`
- `activation='tanh'`
- `recurrent_activation='sigmoid'`
- `return_sequences=True` so attention sees every step

Adam uses `lr=` (Keras 2 name). On Keras 3 / TF 2.16+ this constructor keyword is `learning_rate`.

## Attention layer

`attention_layer.py` is a Keras `Layer` following Raffel et al. 2015 ([arXiv:1512.08756](https://arxiv.org/abs/1512.08756)):

- Input `(samples, steps, features)`.
- Trainable `W` of shape `(features,)`.
- Optional bias `b` of shape `(steps,)`.
- Score `tanh(x W + b)`, softmax over steps (with `epsilon` in the denominator), weighted sum.

Masking is supported but `compute_mask` returns `None`, so the Dense layer does not see a mask. Padding tokens still get a score; the model has to learn to down-weight them.

`examples/attention_numpy.py` is a line-for-line numpy port used by `examples/run_attention.py` and the unit tests.

## What was not trained

The notebooks do not search:

- LSTM width (256 is fixed)
- number of stacked Bi-LSTMs (2 is fixed)
- dropout (0.25 / 0.4)
- learning rate (0.001)
- tree depth / `n_estimators` / SVM `C`

So the result tables compare *default sklearn vs one LSTM recipe*, not tuned champions of each family. That is acceptable for a "does emoji help" ablation. It is not a claim that random forest beats a tuned GBT, or that 256 is the right hidden size.

## Inference API (original)

Classical:

```python
from data_utils import ml_read_data
X, y, X_emoji, y_emoji = ml_read_data(sent, lab, glove, emoji2vec)
model.predict(X)          # single-modal
model.predict(X_emoji)    # multi-modal
```

Deep:

```python
from data_utils import ReadOpen, Preprocess, preprocess_test
from dl_model import PrepModel
docs, labels, n = ReadOpen(sent, lab)
ids, E, maxlen, tok = Preprocess(docs, n, glove, emoji2vec, get_emoji2vec=True)
model = PrepModel(n, E, maxlen)
model.fit(ids, labels, ...)
```

The examples package does not call these, because they import Keras / Gensim. It reimplements the feature math and a tiny logistic regressor so the same story runs with numpy.
