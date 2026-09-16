# Project overview

This repository is a **personal** University of Copenhagen Computational
Cognitive Science 2 (2023) final project. It studies whether emoji
co-occurrence helps Twitter sarcasm detection when the text stream already
contains hashtags, mentions, and other social cues.

The original experiment compared:

- classical classifiers on mean-pooled embeddings
- a bidirectional LSTM with a temporal attention layer

and ran both families in a **single-modal** setting (word embeddings only)
and a **multi-modal** setting (word embeddings plus emoji2vec).

Nothing here is company work. The code, notebooks, and the documentation in
`docs/` plus the walkthroughs in `examples/` exist so the 2023 course project
can be read, re-run in pieces, and reused as a personal learning archive.

## Research question

Sarcasm on Twitter is often marked by a mismatch between literal wording and
intended meaning. People resolve that mismatch with extra-linguistic signals:

- discourse markers such as `#not`, `#sarcasm`, `#yeahright`
- emoji that invert or exaggerate polarity (`😒`, `👌`, `😅`)
- elongated spellings and punctuation (`loovee`, `...`, `!!!`)

The project asks a concrete modeling question:

> If a tweet is represented with GloVe Twitter word vectors, does concatenating
> an emoji2vec average improve sarcasm classification, and does that gain
> survive when the evaluation set is restricted to tweets that actually contain
> emoji?

## Short answer from the 2023 runs

On the balanced 2,000-tweet test split, a Bi-LSTM + attention model was the
strongest system. Adding emoji2vec improved test accuracy from **86.35%** to
**87.35%**. The gain was larger on the 278-tweet emoji-only subtest:
**86.69% → 89.21%** accuracy and **89.40% → 91.07%** F1.

Among the classical baselines, Random Forest was the only model that clearly
benefited from the concatenated emoji channel on the subtest (80.58% → 85.25%
accuracy). SVM slightly *lost* accuracy on the full test set when emoji
vectors were added, which is discussed in [experiments.md](experiments.md).

## What lives in this repository

| Path | Role |
| --- | --- |
| `dataset/` | Train / test / emoji-subtest sentence and label CSVs |
| `data_utils.py` | Tweet reading, mean pooling, Keras tokenizer + embedding matrix |
| `dl_model.py` | Bi-LSTM + attention classifier used in the original notebooks |
| `attention_layer.py` | Raffel-style temporal attention layer (Keras) |
| `baseline_models.ipynb` | SVM, decision tree, random forest, gradient boosting |
| `evaluate_loaded_dl_models.ipynb` | Reload saved Keras models and score them |
| `get_metrics_of_models.ipynb` | Accuracy / F1 / precision / recall plus comparison plots |
| `baseline_models/` | Pickled classical models from the 2023 run |
| `model/` | Saved Keras graphs for the single- and multi-modal Bi-LSTMs |
| `emoji2vec.bin`, `emoji2vec_twitter.bin` | Pretrained emoji vector tables |
| `docs/` | Architecture, data, experiment, and reproduction notes |
| `examples/` | Runnable walkthroughs that do **not** need GloVe or TensorFlow |

The large GloVe Twitter 27B 200-d table used in 2023 is **not** checked in.
The Keras `variables/` shards under `model/` are also absent, so the saved
models cannot be loaded as-is. See [reproduction.md](reproduction.md).

## Two modeling stacks

```
raw tweet CSV
      │
      ▼
NLTK TweetTokenizer + lowercasing          (data_utils.ReadOpen)
      │
      ├──────────────────────────┐
      ▼                          ▼
mean GloVe (+ mean emoji2vec)    Keras Tokenizer → padded ids
      │                          │
      ▼                          ▼
SVM / DT / RF / GBT              frozen Embedding → 2× Bi-LSTM
                                 → Attention → Dense(sigmoid)
```

The classical path compresses a variable-length tweet into one 200-d or
400-d vector. The neural path keeps token order and lets attention reweight
timesteps before the final logistic unit.

## Document map

1. [data-pipeline.md](data-pipeline.md) — splits, label layout, tokenization, embeddings
2. [architecture.md](architecture.md) — attention math and the Keras model
3. [experiments.md](experiments.md) — recorded metrics and how to read them
4. [reproduction.md](reproduction.md) — environment, missing files, how to rerun
5. [notebook-map.md](notebook-map.md) — what each original notebook actually does
6. [glossary.md](glossary.md) — terms used in the 2023 write-up
7. [../examples/README.md](../examples/README.md) — scripts you can run today
