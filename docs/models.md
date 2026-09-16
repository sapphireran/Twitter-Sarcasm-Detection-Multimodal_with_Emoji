# Models

Four classical classifiers and one recurrent architecture were trained in
single-modal (`W`) and multi-modal (`WE`) versions. The constructors live in
the 2023 notebooks; this page records what those notebooks actually fit.

## Classical baselines

`baseline_models.ipynb` tries to reload a pair of joblib files per family and
only trains if the files are missing. The directory that shipped with the
repo currently contains Decision Tree and Gradient Boosting pickles only:

```
baseline_models/dt_classifier.pkl
baseline_models/dt_classifier_we.pkl
baseline_models/gbt_classifier.pkl
baseline_models/gbt_classifier_we.pkl
```

SVM and Random Forest pickles are referenced by the notebook
(`svm_classifier.pkl`, `rf_classifier.pkl`, and the `_we` variants) but are
not in git. `get_metrics_of_models.ipynb` looks for a third naming scheme
(`svm_model.pkl`). If you want to regenerate the classical table you will
need GloVe plus a retrain; the published numbers are copied into
`docs/results.md` from the executed notebook outputs.

Hyperparameters in the training cells are library defaults
(`SVC()`, `DecisionTreeClassifier()`, `RandomForestClassifier()`,
`GradientBoostingClassifier()`). A few `except FileNotFoundError` branches
accidentally construct the wrong class (the RF single-modal fallback fits an
`SVC`, the DT multi-modal fallback also fits an `SVC`). Those branches only
run when the pickle is absent, so they did not produce the saved DT/GBT
files.

## Bi-LSTM + attention

`dl_model.PrepModel(count, embedding_matrix, l, lrate=0.001)` builds:

* frozen `Embedding(count, 200, input_length=l)`
* `Dropout(0.25)`
* `Bidirectional(LSTM(256, he_normal, tanh, sigmoid recurrent, return_sequences=True))`
* `Dropout(0.4)`
* a second identical Bi-LSTM
* `Dropout(0.4)`
* `Attention()`
* `Dense(1, sigmoid)`
* `Adam(lr=0.001)`, `binary_crossentropy`, accuracy

The two shipped SavedModel directories are:

| Path | `get_emoji2vec` | Notebook name |
| --- | --- | --- |
| `model/best_model_single_modal` | `False` | Bi-LSTM+ATT_w |
| `model/best_model_multi_modal` | `True` | Bi-LSTM+ATT_we |

`evaluate_loaded_dl_models.ipynb` reports 2,510,848 parameters for both. The
embedding table is stored inside the SavedModel; the `trainable=False` flag
from `PrepModel` is what the training script used. The summary printed after
a TF2 reload wraps several layers as `ModuleWrapper`, which is a loading
artifact, not a second architecture.

## Attention layer details

`attention_layer.Attention` is a Keras 2 custom layer:

* `supports_masking = True`
* `W` is glorot-uniform, shape `(features,)`
* optional `b` is zeros, shape `(steps,)`
* scores go through `tanh` then `exp`
* the mask (if any) multiplies the unnormalized weights
* the denominator is `sum(weights) + epsilon` to avoid NaNs early in training
* the layer returns the weighted sum, shape `(batch, features)`, and does
  **not** forward the mask

`examples/attention_walkthrough.py` rebuilds that pooling in NumPy on a toy
three-token tweet so you can see the `α` vector without TensorFlow.

## Heuristic baseline (docs only)

`sarcasm_lib.heuristic` is not one of the course models. It scores explicit
cues so the write-up can separate "the label is written in a hashtag" from
"the network understood the tweet". Default rule: predict sarcastic if the
score is at least 1.0. Marker hashtags contribute 2.0; a positive opener plus
a negative emoji contributes 1.0.

Use it as a ceiling on how far surface cues go, not as a submitted system.
