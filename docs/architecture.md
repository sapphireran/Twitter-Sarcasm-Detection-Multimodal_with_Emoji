# Architecture

Two families of models were trained for the course project.

## Classical baselines

Implemented in `baseline_models.ipynb`. Each tweet becomes a 200-d mean
GloVe vector (single-modal) or a 400-d concat of mean GloVe and mean
emoji2vec (multi-modal). Classifiers:

* SVM (`sklearn.svm.SVC`)
* Decision tree
* Random forest
* Gradient boosting

Pickled trees/boosters still sit in `baseline_models/`
(`dt_classifier.pkl`, `dt_classifier_we.pkl`, `gbt_classifier.pkl`,
`gbt_classifier_we.pkl`). The SVM and random-forest pickles referenced
in the metrics notebook are not all present in this checkout.

## BiLSTM + attention (best model)

Defined in `dl_model.py` / `attention_layer.py`. Saved copies:

* `model/best_model_single_modal`
* `model/best_model_multi_modal`

`evaluate_loaded_dl_models.ipynb` prints both as `sequential_*` with:

| Layer | Output | Notes |
| --- | --- | --- |
| Embedding 200-d, frozen | `(None, 78, 200)` | GloVe / mixed emoji table from `Preprocess` |
| Dropout 0.25 | same | |
| Bidirectional LSTM 256 | `(None, 78, 512)` | `return_sequences=True`, He init, tanh / sigmoid |
| Dropout 0.4 | | |
| Bidirectional LSTM 256 | `(None, 78, 512)` | second layer also returns sequences |
| Dropout 0.4 | | |
| Attention | `(None, 512)` | Raffel 2016, Keras 2 |
| Dense sigmoid | `(None, 1)` | binary cross-entropy, Adam `lr=0.001` |

Trainable weights ≈ 2.51M. Sequence length 78 is the padded train
maximum from `data_utils.Preprocess`.

```
tweet tokens  →  frozen 200-d embedding
              →  dropout
              →  BiLSTM 256 (seq)
              →  dropout
              →  BiLSTM 256 (seq)
              →  dropout
              →  attention pool
              →  sigmoid
```

Single- vs multi-modal here is **not** a second encoder. Both use the
same graph. The difference is how `Preprocess` fills the embedding
matrix: words from GloVe, and (multi-modal only) emoji tokens averaged
from emoji2vec when GloVe misses them. See [methodology.md](methodology.md).

## Attention math

`attention_layer.Attention` follows Raffel et al.,
[Feed-Forward Networks with Attention](https://arxiv.org/abs/1512.08756):

1. `e_t = tanh(x_t · W + b_t)` with `W` of size `features` and `b` of
   size `timesteps`
2. `a = softmax(e)` after masking, plus `epsilon` in the denominator
3. output `∑_t a_t x_t`

`sarcasm_toolkit.attention.attention_pool` reimplements that in pure
Python. `examples/03_attention_pooling.py` uses 8-d toy vectors so you
can watch `#not` and emoji attract mass without TensorFlow.

## Cue baselines (docs/examples only)

These are **not** part of the 2023 grade; they document the dataset
floor:

* `LexiconBaseline` — sarcastic iff a cue hashtag is present
* `CueLogistic` — logistic regression on 17 surface features (hashtags,
  emoji counts, elongation, `love when` frames, …)

They live in `sarcasm_toolkit/baseline.py` and are exercised by
`examples/04_lexicon_baseline.py` and `examples/05_cue_logistic.py`.
