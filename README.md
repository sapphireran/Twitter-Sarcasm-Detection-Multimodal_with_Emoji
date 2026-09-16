# Multimodal Twitter sarcasm detection (emoji + text)

Personal final project for **Computational Cognitive Science 2**, University of
Copenhagen, 2023. The models ask whether emoji2vec adds anything on top of
GloVe-Twitter when the task is binary sarcasm detection.

This is a personal course repository. It is not company code.

## Headline numbers

From the shipped checkpoints, evaluated in `get_metrics_of_models.ipynb`:

| Model | Test accuracy | Test F1 | Subtest accuracy | Subtest F1 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest, words + emoji | 0.818 | 0.826 | 0.853 | 0.884 |
| Bi-LSTM + attention, words only | 0.8635 | 0.866 | 0.867 | 0.894 |
| Bi-LSTM + attention, words + emoji | **0.8735** | **0.869** | **0.892** | **0.911** |

Subtest is not a second domain. It is the 278 emoji-bearing tweets inside the
2,000-tweet official test set, and it is saturated with `#not` /
`#sarcastictweet`. Quote the **test** column. Details and the explicit-cue
heuristic that reaches 0.81 accuracy on test (precision 1.0, recall 0.62) are
in [`docs/results.md`](docs/results.md) and [`docs/dataset.md`](docs/dataset.md).

## What is in this repo

```
attention_layer.py     Raffel attention (Keras)
data_utils.py          Original loaders + embedding matrix
dl_model.py            Bi-LSTM constructor
baseline_models/       Decision Tree and GBT pickles
model/                 Saved single-modal and multi-modal Bi-LSTMs
dataset/               train / test / subtest tweets and labels
emoji2vec*.bin         emoji2vec tables (GloVe is not included)
sarcasm_lib/           Dependency-light helpers for the examples
examples/              Runnable dataset, heuristic, and attention scripts
docs/                  Architecture, data, models, results, reproduction
tests/                 Unit tests that hit the real CSV files
```

GloVe-Twitter 27B 200d is **not** checked in. The original notebooks need it;
the example scripts do not.

## Docs

| Page | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | W vs WE pipelines, why subtest is easy |
| [docs/dataset.md](docs/dataset.md) | Split sizes, `#not` leakage, emoji rates |
| [docs/preprocessing.md](docs/preprocessing.md) | `ReadOpen`, mean pooling, embedding fill-in |
| [docs/embeddings.md](docs/embeddings.md) | GloVe vs the two shipped emoji2vec tables |
| [docs/models.md](docs/models.md) | Baselines, Bi-LSTM, attention, heuristic |
| [docs/results.md](docs/results.md) | Accuracy / F1 / P / R tables |
| [docs/reproduction.md](docs/reproduction.md) | How to rerun examples vs notebooks |
| [docs/examples.md](docs/examples.md) | Script index |

## Examples (no TensorFlow)

```bash
python3 -m pip install -r requirements-examples.txt
python3 -m unittest discover -s tests -v

python3 examples/inspect_dataset.py
python3 examples/tokenize_tweets.py --split train
python3 examples/heuristic_baseline.py
python3 examples/attention_walkthrough.py
python3 examples/predict_cli.py "I love walking to school #not"
python3 examples/predict_cli.py --file examples/sample_tweets.txt
```

`examples/inspect_emoji2vec.py` is optional and needs Gensim. It only reads
the `emoji2vec_twitter.bin` file that is already in the tree.

## Original notebooks

1. Drop GloVe at `glove.twitter.27B.200d.bin` (or `glove_tt.txt` for the
   metrics notebook).
2. Point the baseline notebook at `dataset/*.csv` (it currently expects the
   CSV files in the working directory).
3. Open `baseline_models.ipynb`, `evaluate_loaded_dl_models.ipynb`, then
   `get_metrics_of_models.ipynb`.

See [docs/reproduction.md](docs/reproduction.md) for the missing SVM / RF
pickles, the `Adam(lr=...)` rename, and the unseeded shuffle in
`ml_read_data`.

## License

MIT. See `LICENSE`.
