# Twitter sarcasm detection with emoji embeddings

Personal final project for **UCPH Computational Cognitive Science 2 (2023)**.
The question was narrow: if you already have GloVe Twitter word vectors, does
adding emoji2vec help detect sarcasm on tweets?

This clone now has two layers:

1. The original notebooks and training code (`data_utils.py`, `dl_model.py`,
   `attention_layer.py`, `*.ipynb`).
2. A new `docs/` + `examples/` tree that runs on the checked-in CSVs with only
   NumPy. Use that if you want to inspect the data, the attention formula, or a
   tiny lexical baseline without reconstructing the 2023 GPU environment.

## Result in one paragraph

A bidirectional LSTM with Raffel attention reached **86.35%** test accuracy
from GloVe alone and **87.35%** once emoji2vec filled in unknown tokens. On the
278-tweet emoji subtest the same pair is **86.69% → 89.21%**. Random forest is
the strongest classical baseline (81.45% / 81.80% on test). Emoji fusion is
real, but it is easiest to see when the tweet actually contains emoji. Full
tables: [docs/results.md](docs/results.md).

## Repository map

```text
dataset/                  train / test / subtest sentence+label CSVs
attention_layer.py        Keras temporal attention (Raffel et al., 2016)
data_utils.py             ReadOpen, mean-pooling, embedding matrix
dl_model.py               Bi-LSTM + attention classifier
baseline_models.ipynb     SVM, DT, RF, GBT on mean vectors
get_metrics_of_models.ipynb   accuracy / F1 / plots from 2023
evaluate_loaded_dl_models.ipynb
emoji2vec*.bin            emoji embeddings (GloVe Twitter is not in git)
docs/                     methodology, dataset, results, reproduction
examples/                 scripts that run on this clone
tests/                    unittest coverage for the examples
```

## Data

| Split | Tweets | Sarcastic | Notes |
| --- | ---: | ---: | --- |
| train | 39,780 | 18,488 | near-balanced |
| test | 2,000 | 1,000 | official evaluation |
| subtest | 278 | 172 | every row is also in test; almost all have emoji |

Labels are `1` sarcastic, `0` not. `ReadOpen` does not use a CSV parser; it
replaces commas with spaces and tokenizes with NLTK `TweetTokenizer`. More
counts, cue rates, and overlap notes: [docs/dataset.md](docs/dataset.md).

## Models

**Classical.** Mean GloVe vector (200-d) vs concatenated GloVe + emoji2vec
(400-d), then SVM / decision tree / random forest / gradient boosting.

**Deep.** Frozen 200-d embedding, dropout, two bidirectional LSTM layers
(256 units each), attention, sigmoid. Multi-modal training writes emoji2vec
rows into the embedding matrix for tokens GloVe missed.

**Examples.** A 19-feature logistic model over `#not`, polarity, contrast,
elongation, and punctuation. Classroom sanity check only.

Architecture write-up: [docs/methodology.md](docs/methodology.md).

## Run the new examples

Needs Python 3.10+ and NumPy (`pip install -r requirements-examples.txt`).

```bash
python3 -m examples.dataset_report --markdown
python3 -m examples.emoji_signals --split test
python3 -m examples.tweet_tokenizer_demo
python3 -m examples.attention_demo
python3 -m examples.metrics_table --metric accuracy
python3 -m examples.lexical_baseline_demo --train-limit 8000
python3 -m unittest discover -s tests
```

Command-by-command notes: [docs/examples.md](docs/examples.md).

## Reproducing the 2023 notebooks

You will need `glove.twitter.27B.200d.bin`, TensorFlow / Keras, gensim 3.x,
NLTK, and the missing sklearn pickles or a full retrain. The SavedModel
folders under `model/` currently contain only `keras_metadata.pb`. Read
[docs/reproduction.md](docs/reproduction.md) before spending time on that path.

`requirements.txt` lists the original notebook stack. `requirements-examples.txt`
is enough for `examples/` and `tests/`.

## License

MIT. Original copyright line is `pang990801` (2023).
