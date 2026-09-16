# Multi-modal Twitter sarcasm detection (emoji + text)

Personal final project for **UCPH Computational Cognitive Science 2 (2023)**.
The question was whether emoji co-occurrence adds anything once you already
have Twitter-tuned word vectors — and whether a small BiLSTM with attention
beats the usual sklearn baselines on the same dump.

This repository keeps the original training / evaluation notebooks, the
course data split, and (now) a set of **dependency-light docs and examples**
that replay the pipeline without the 200-d GloVe download.

> Personal / course work only. Nothing here is product or company code.

## Result in one paragraph

On the official 2,000-tweet test set the best model is a two-layer
**bidirectional LSTM + attention** with a frozen 200-d embedding matrix:
**86.35%** accuracy from GloVe alone, **87.35%** when emoji tokens fall
back to emoji2vec. The same emoji signal is much louder on a 278-tweet
emoji-rich subtest (**89.21%**). Random forest is the strongest classical
baseline (81.5% / 81.8%). A transparent `#not` rule is precise but covers
only a minority of sarcastic tweets — the network is not just memorising
hashtags.

Full tables: [docs/results.md](docs/results.md).

## Repository map

```
attention_layer.py     Raffel-style Keras attention used on top of BiLSTM
data_utils.py          ReadOpen, GloVe / emoji2vec averaging, Keras padding
dl_model.py            BiLSTM + Attention builder (Adam, binary cross-entropy)
dataset/               train / test / subtest sentence+label CSVs
model/                 saved Keras graphs (weights folders were not uploaded)
baseline_models/       pickled DT / GBT (SVM / RF pickles were not uploaded)
baseline_models.ipynb  sklearn baselines, single- vs multi-modal
get_metrics_of_models.ipynb   accuracy / F1 / precision / recall export
evaluate_loaded_dl_models.ipynb  reload the two best Keras runs
docs/                  dataset card, architecture, results, reproduction
examples/              runnable walk-throughs (NumPy + stdlib only)
tests/                 unittest coverage for the example library
```

## Data

| Split | Rows | Sarcastic | Notes |
| --- | ---: | ---: | --- |
| `dataset/train_*.csv` | 39,780 | 46.5% | slightly majority non-sarcastic |
| `dataset/test_*.csv` | 2,000 | 50.0% | official balanced test |
| `dataset/subtest_*.csv` | 278 | 61.9% | emoji-heavy slice used as a stress test |

Labels are `0` / `1`. Sentences are stored one tweet per line; some rows
are CSV-quoted because the original dump had commas inside the text.
`ReadOpen` historically split those commas back into spaces — see
[docs/preprocessing.md](docs/preprocessing.md).

Explore without GloVe:

```bash
python3 examples/explore_dataset.py
```

## Models

**Classical (mean-pooled vectors → sklearn)**

* word-only: 200-d mean of Twitter GloVe
* multi-modal: that 200-d mean concatenated with a 200-d mean of emoji2vec
* SVM, decision tree, random forest, gradient boosting

**Deep (sequence model)**

* frozen embedding matrix (GloVe, with an emoji2vec fallback per token)
* dropout → BiLSTM(256) → dropout → BiLSTM(256) → Attention → sigmoid
* trained with Adam (`lr=0.001`) and binary cross-entropy

Architecture notes and a NumPy replay of the attention layer:
[docs/architecture.md](docs/architecture.md),
`python3 examples/attention_numpy.py`.

## Examples (no GloVe / TensorFlow)

```bash
python3 examples/explore_dataset.py
python3 examples/tokenize_tweets.py --split subtest --n 4
python3 examples/sarcasm_cues.py --split test
python3 examples/attention_numpy.py
python3 examples/embedding_pipeline.py --split subtest --n 6
python3 examples/report_results.py --all
python3 -m unittest tests.test_examples
```

Details: [examples/README.md](examples/README.md).

## Original notebooks (need GloVe + Keras 2)

The 2023 notebooks expect:

* `glove.twitter.27B.200d.bin` (or `glove_tt.txt` in the metrics notebook)
* `emoji2vec_twitter.bin` (already in this repo)
* TensorFlow 2 / Keras 2, `gensim`, `nltk`, `emoji`, `scikit-learn`

Those binaries and the pickled SVM / RF files are **not** all present in
the GitHub snapshot. See [docs/reproduction.md](docs/reproduction.md) for
what is missing and how the recorded numbers were produced.

## Docs

* [docs/dataset.md](docs/dataset.md) — split card, label rates, cue stats
* [docs/preprocessing.md](docs/preprocessing.md) — `ReadOpen` to embedding matrix
* [docs/architecture.md](docs/architecture.md) — BiLSTM + attention, baselines
* [docs/results.md](docs/results.md) — recorded metrics and what they mean
* [docs/reproduction.md](docs/reproduction.md) — how to rerun, what is missing
* [docs/experiment-notes.md](docs/experiment-notes.md) — personal 2023 notes
* [docs/code-map.md](docs/code-map.md) — file-by-file reading order

## License

MIT. Copyright (c) 2023 pang990801 / Sapphire Ran.
