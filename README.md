# Twitter sarcasm detection with text + emoji

Personal final project for **Computational Cognitive Science 2**, University of
Copenhagen, 2023. The question is whether a tweet-level sarcasm detector
improves when the emoji channel is modelled explicitly, instead of treating
pictographs as out-of-vocabulary characters that collapse to a zero vector.

This repository is a personal academic archive (MIT license, 2023). It is not
affiliated with any employer.

## What is in the box

| Path | Role |
| --- | --- |
| `dataset/` | Train / test / emoji-only subtest CSVs (39,780 / 2,000 / 278 tweets) |
| `data_utils.py` | Tweet tokenization, GloVe + emoji2vec lookup, average pooling |
| `dl_model.py` | Frozen-embedding BiLSTM + Raffel attention, sigmoid output |
| `attention_layer.py` | Keras 2 attention layer (Raffel et al. 2016) |
| `baseline_models/` | Sklearn pickles (decision tree, gradient boosting; SVM/RF were not committed) |
| `model/` | SavedModel stubs for the best single-modal and multimodal nets |
| `emoji2vec_twitter.bin` | 1,661 × 200 emoji2vec, aligned with GloVe-Twitter 200d |
| `emoji2vec.bin` | Original 1,661 × 300 emoji2vec (Eisner et al. 2016) |
| `docs/` | Dataset, architecture, results, and reproduction notes |
| `examples/` | Runnable scripts that do **not** need TensorFlow or GloVe |

The 2023 notebooks (`baseline_models.ipynb`, `get_metrics_of_models.ipynb`,
`evaluate_loaded_dl_models.ipynb`) are the original experiment log. Prefer the
`examples/` scripts if you just want to inspect the CSVs or the emoji table.
A later control (`examples/03_lexical_cues.py`) shows that a `#sarcasm` /
`#not` rule already scores 80.7% on this test split — read
[`docs/08-control-experiments.md`](docs/08-control-experiments.md) before
treating the 87% neural figure as in-the-wild performance.

## Headline numbers (held-out)

Reported in `get_metrics_of_models.ipynb` after loading the trained artifacts.
**W** = word / GloVe only. **WE** = word + emoji2vec. **subtest** is the 278
tweets that contain at least one emoji.

| Model | Test acc W | Test acc WE | Subtest acc W | Subtest acc WE |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 72.65 | 72.95 | 77.70 | 79.86 |
| SVM | 76.90 | 76.30 | 81.29 | 82.37 |
| Gradient boosting | 74.60 | 74.75 | 79.50 | 79.50 |
| Random forest | 81.45 | 81.80 | 80.58 | 85.25 |
| BiLSTM + attention | **86.35** | **87.35** | **86.69** | **89.21** |

The multimodal lift is small on the full test set (most tweets have no emoji)
and larger on subtest. That is the experimental design, not an accident: see
[`docs/02-dataset.md`](docs/02-dataset.md) and [`docs/04-results.md`](docs/04-results.md).

## Two modelling stacks

**Classical (tweet-level average vectors).** Each tweet becomes the mean of the
GloVe-Twitter 200d rows that hit. The multimodal variant concatenates that 200-d
mean with the mean of the emoji2vec rows, producing a 400-d vector. SVM,
decision trees, random forests, and gradient boosting are trained on those
vectors (`ml_read_data` in `data_utils.py`).

**Neural (token sequence + attention).** Tokens are mapped through a frozen
embedding matrix. Rows come from GloVe when the token is in-vocabulary, and
from the average of emoji2vec rows when `emoji.emoji_list` can parse the
token. Two stacked bidirectional LSTMs (256 units each direction) emit a
sequence; Raffel attention pools it; a sigmoid dense layer predicts sarcasm.
`get_emoji2vec=False` zeros the emoji rows and is the single-modal control.

```
tweet tokens
    │
    ▼
frozen 200-d embedding  (GloVe and/or emoji2vec)
    │
    ▼
Dropout 0.25
    │
    ▼
BiLSTM 256  →  Dropout 0.4
    │
    ▼
BiLSTM 256  →  Dropout 0.4
    │
    ▼
Attention (Raffel)  →  Dense(1, sigmoid)
```

## Quick start (docs / examples, no GloVe)

The example scripts only need the Python standard library and `numpy`, which
is enough to explore the CSVs, read emoji2vec, and run a bag-of-words control.

```bash
python3 examples/01_explore_dataset.py
python3 examples/02_emoji_cooccurrence.py --split train
python3 examples/03_lexical_cues.py
python3 examples/04_inspect_emoji2vec.py
python3 examples/05_toy_attention.py
python3 examples/06_bow_baseline.py
python3 examples/07_average_pooling.py --real-emoji2vec
python3 examples/08_results_table.py
python3 -m unittest discover -s tests -v
```

## Re-running the 2023 notebooks

You need:

1. Python 3.8–3.10, TensorFlow 2.x with Keras 2, Gensim 3 or 4, NLTK, `emoji`,
   scikit-learn, joblib.
2. `glove.twitter.27B.200d` converted to word2vec binary as
   `glove.twitter.27B.200d.bin` (not shipped; ~1.2M rows × 200).
3. NLTK `punkt` is **not** required; `TweetTokenizer` is rule-based.

Then open `baseline_models.ipynb` or `evaluate_loaded_dl_models.ipynb`. Paths
inside the notebooks sometimes assume the CSVs live in the working directory
rather than `dataset/`; `get_metrics_of_models.ipynb` uses the `dataset/`
prefix. See [`docs/05-reproduction.md`](docs/05-reproduction.md) for the
gotchas (SavedModel wrappers, Gensim `vocab` vs `key_to_index`, Adam `lr`).

## Project layout (code)

```
data_utils.py          ReadOpen, AverageVectorPerTweet, Preprocess
dl_model.py            PrepModel (BiLSTM + Attention)
attention_layer.py     Keras Layer, masking, tanh scores
baseline_models/       *.pkl for DT / GBT (single- and multi-modal)
model/best_model_*     Keras SavedModel directories
docs/                  long-form notes
examples/              numbered scripts + examples/common/
tests/                 offline unit tests for the example library
```

## Citation / provenance

* Course: Computational Cognitive Science 2, UCPH, 2023.
* Attention: Raffel & Ellis, *Feed-Forward Networks with Attention Can
  Solve Some Long-Term Memory Problems*, 2016, <https://arxiv.org/abs/1512.08756>.
* emoji2vec: Eisner, Rocktäschel, Augenstein, Bošnjak, Riedel, *emoji2vec:
  Learning Emoji Representations from their Description*, 2016,
  <https://arxiv.org/abs/1609.08359>.
* Word vectors: GloVe Twitter 27B, 200d (Pennington, Socher, Manning).

Tweets in `dataset/` are a course-provided export with `<user>` placeholders.
Do not treat them as a newly collected corpus and do not try to re-identify
authors.

## License

MIT. See `LICENSE`.
