# Architecture

## Classical heads

`baseline_models.ipynb` trains one W model and one WE model for each
of SVM, decision tree, random forest, and gradient boosting. Features
are the 200d or 400d averaged vectors from `ml_read_data`. Defaults
are sklearn defaults (no grid search in the notebook).

Checked-in pickles:

| File | Present? |
| --- | --- |
| `baseline_models/dt_classifier.pkl` | yes |
| `baseline_models/dt_classifier_we.pkl` | yes |
| `baseline_models/gbt_classifier.pkl` | yes |
| `baseline_models/gbt_classifier_we.pkl` | yes |
| SVM / RF pickles | **no** |

If a pickle is missing, the notebook retrains. Several `except`
branches construct the wrong class:

* DT-WE falls back to `SVC()`
* RF-W falls back to `SVC()`
* GBT-W falls back to `SVC()`

The recorded 2023 scores were produced from already-saved pickles, so
those branches did not run in the published outputs. They will bite
anyone who retrains from this clone.

## BiLSTM + Raffel attention

`dl_model.PrepModel` builds:

```text
Embedding(count, 200, trainable=False, input_length=L)
Dropout(0.25)
Bidirectional(LSTM(256, return_sequences=True))
Dropout(0.4)
Bidirectional(LSTM(256, return_sequences=True))
Dropout(0.4)
Attention()          # Raffel 2015, see attention_layer.py
Dense(1, sigmoid)
Adam(lr=0.001), binary_crossentropy, accuracy
```

Each bidirectional LSTM emits 512 units (256 forward + 256 backward).
`evaluate_loaded_dl_models.ipynb` reports 2,510,848 parameters and a
length-78 sequence. The embedding weights are stored inside the
SavedModel, not as a separate `.npy`.

## Attention layer details

`attention_layer.Attention` is the Keras 2 implementation used in
2016–2018 sentiment papers:

* score `e_t = tanh(h_t · W + b_t)`
* `W` has shape `(features,)` — a dot-product head, not a matrix
* `b` has shape `(timesteps,)` when `bias=True`
* mask is applied after `exp`, then the row is renormalized with
  `K.epsilon()` in the denominator

The timestep-shaped bias is the surprising bit. It means the layer
learns a positional preference over the *padded* length 78, not a
content-only score. The NumPy port in `ccs2lab.attention` keeps that
shape so a walkthrough can show positional bias and masking
separately.

## What “multi-modal” is not

There is no second encoder, no cross-attention between words and
emoji, and no character-level emoji model. WE is early fusion:
average the emoji table, or splice emoji rows into the word
embedding matrix. That is enough for a CCS2 final and too little if
you want a modern multi-modal claim.
