# Architecture

The neural classifier is a small Keras `Sequential` defined in `dl_model.py.PrepModel`. Attention is the custom layer in `attention_layer.py`, following Raffel et al., *Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems* ([arXiv:1512.08756](https://arxiv.org/abs/1512.08756)).

## Graph

```text
token ids (batch, 78)
        │
        ▼
Embedding(vocab_rows, 200, trainable=False)   ← GloVe / emoji2vec table
        │
        ▼
Dropout(0.25)
        │
        ▼
Bidirectional LSTM(256, return_sequences=True)  → (batch, 78, 512)
        │
        ▼
Dropout(0.4)
        │
        ▼
Bidirectional LSTM(256, return_sequences=True)  → (batch, 78, 512)
        │
        ▼
Dropout(0.4)
        │
        ▼
Attention()   # Raffel feed-forward attention over time
        │
        ▼
Dense(1, sigmoid)   # P(sarcastic)
```

Optimizer: Adam at `lr=0.001`. Loss: binary cross-entropy. Metric: accuracy.

The saved-model summaries in `evaluate_loaded_dl_models.ipynb` report **2,510,848** parameters, all trainable at load time. That number is dominated by the two bidirectional LSTMs (`935,936` + `1,574,912`). The frozen embedding table is stored as weights but wrapped in `ModuleWrapper` layers in the recorded summary, so the printed “trainable” count there is not a reliable guide to whether GloVe was actually updated.

## Attention layer

Given a sequence `x` of shape `(samples, steps, features)`:

1. Score each step: `e_t = tanh(x_t · W + b_t)`  
   `W` is a vector of size `features`. `b` is a **per-timestep** bias of size `steps` (78 in the trained models).
2. Softmax over time, with mask support and `epsilon` in the denominator to avoid NaNs.
3. Return the weighted sum over time: `(samples, features)`.

Because `b` is length `steps`, the layer assumes a **fixed** sequence length after padding. That matches `pad_sequences` + `input_length=l`, but it is not length-agnostic. A tweet padded from 12 tokens to 78 still has 66 bias slots that only ever see pad positions (the mask, when present, zeros those attentions).

`compute_mask` returns `None`, so later layers do not inherit the padding mask. Only the attention softmax uses it.

A NumPy walk-through of the same equations lives in `examples/attention_walkthrough.py` and `examples/sarcasm_lab/attention.py`.

## Why bidirectional LSTM + attention for sarcasm

Sarcasm on Twitter is often a **late cue** (`#not`, a frowning emoji, a reversal after a positive opener). A unidirectional LSTM can bury that cue; bidirectional states plus attention let the classifier put weight on the reversal without hand-coding the hashtag.

That is also why the multimodal table helps most when the cue is an emoji that GloVe never saw as a word. Substituting emoji2vec for those rows gives the LSTMs a nonzero channel for `😒` / `😅` instead of a zero row.

## Classical baselines

`baseline_models.ipynb` trains four sklearn estimators on the **mean-pooled** 200-D or 400-D vectors:

- SVM (`SVC` defaults)
- Decision tree
- Random forest
- Gradient boosting

Each estimator is saved twice (`*_classifier.pkl` and `*_classifier_we.pkl`). Only some of those pickles are still in `baseline_models/` (decision tree and GBT). SVM and random-forest pickles are referenced by the notebooks but are not in the tree anymore.

There is a copy-paste bug in the “train if missing” branches: the multimodal decision-tree fit instantiates `SVC()`, and the single-modal random-forest / GBT fits also instantiate `SVC()`. If you retrain from those cells, you will not get the named algorithm. The **loaded** pickle metrics in the notebooks are the ones recorded in [results.md](results.md).

## Saved Keras directories

`model/best_model_single_modal` and `model/best_model_multi_modal` contain `saved_model.pb` and `keras_metadata.pb` only. A full TensorFlow SavedModel normally also has a `variables/` directory. Reload those folders only if you restore the missing variable shards from a local backup; the graphs alone are not enough to `evaluate()`.
