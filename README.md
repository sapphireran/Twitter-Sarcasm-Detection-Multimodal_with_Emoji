# Multi-modal Twitter Sarcasm Detection (Text + Emoji)

Personal final project for **UCPH Computational Cognitive Science 2 (2023)**.

The question this repo answers is simple: *does an emoji vector help a sarcasm classifier that already has Twitter-trained word embeddings?*

The answer from the original experiments is **yes, but mostly on emoji-bearing tweets**. A bidirectional LSTM with attention is the strongest model in the archive. Concatenating GloVe Twitter word averages with emoji2vec averages helps classical baselines on the emoji-heavy **subtest** split more than on the balanced full test set.

This tree now includes:

- the original notebooks, preprocessing, and saved models
- a written map of the architecture, data, and results
- runnable examples that do **not** need GloVe, TensorFlow, or the 200-d binaries

## What is in the box

| Path | Role |
| --- | --- |
| `dataset/` | Train / test / emoji-heavy subtest tweets and 0/1 labels |
| `data_utils.py` | Tweet tokenization, GloVe + emoji2vec lookup, padding |
| `dl_model.py` | BiLSTM + attention classifier (`PrepModel`) |
| `attention_layer.py` | Raffel-style temporal attention (Keras) |
| `baseline_models/` | Pickled Decision Tree and Gradient Boosting pairs |
| `model/` | Saved Keras graphs for single-modal and multi-modal BiLSTMs |
| `*.ipynb` | Training / reload / metric notebooks from the 2023 run |
| `docs/` | Architecture, dataset, methodology, results, reproduction |
| `examples/` | Stdlib + NumPy walkthroughs of the same ideas |
| `tests/` | Unit tests for the example library |

Saved SVM / Random Forest pickles and the GloVe Twitter 27B 200-d file are **not** in git (they were loaded from local paths in the notebooks). See [docs/reproduction.md](docs/reproduction.md).

## Results snapshot (2023 notebooks)

Numbers below are copied from executed notebook outputs, not re-run in this environment.

| Model | Modalities | Test acc | Test F1 | Subtest acc | Subtest F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Decision Tree | word | 0.727 | 0.756 | 0.777 | 0.834 |
| Decision Tree | word + emoji | 0.730 | 0.757 | 0.799 | 0.846 |
| Gradient Boosting | word | 0.746 | 0.751 | 0.795 | 0.838 |
| Gradient Boosting | word + emoji | 0.748 | 0.753 | 0.795 | 0.836 |
| SVM | word | 0.769 | 0.772 | 0.813 | 0.852 |
| SVM | word + emoji | 0.763 | 0.766 | 0.824 | 0.853 |
| Random Forest | word | 0.815 | 0.823 | 0.806 | 0.852 |
| Random Forest | word + emoji | 0.818 | 0.826 | **0.853** | **0.884** |
| BiLSTM + Attention | word | 0.864 | 0.866 | 0.867 | 0.894 |
| BiLSTM + Attention | word + emoji | **0.874** | **0.869** | **0.892** | **0.911** |

The multi-modal BiLSTM is the best overall model. The largest *relative* emoji gain on a classical model is Random Forest on subtest (+4.7 acc, +3.2 F1).

Full tables and caveats: [docs/results.md](docs/results.md).

## Data at a glance

| Split | Rows | Sarcastic | Non-sarcastic | Notes |
| --- | ---: | ---: | ---: | --- |
| `dataset/train_*.csv` | 39,780 | 18,488 | 21,292 | Mentions replaced with `<user>` |
| `dataset/test_*.csv` | 2,000 | 1,000 | 1,000 | Balanced; 48 tweets also appear in train |
| `dataset/subtest_*.csv` | 278 | 172 | 106 | **Exact subset of test** with high-codepoint (emoji-like) characters |

`#not` is a loud lexical cue: 3,105 sarcastic training tweets contain it versus 83 non-sarcastic ones. A cue-only logistic model hits **64.1% test acc / 78.8% subtest acc** — the spelling floor the neural tables have to beat. After `ReadOpen`’s comma flatten, 242 test strings also appear in train; docs call that out instead of hiding it.

Details: [docs/dataset.md](docs/dataset.md).

## Architecture (deep model)

```
tokens ── Tokenizer + pad (post) ──► Embedding(200, frozen GloVe / emoji2vec mix)
                                         │
                                      Dropout 0.25
                                         │
                               BiLSTM(256) return_sequences
                                         │
                                      Dropout 0.40
                                         │
                               BiLSTM(256) return_sequences
                                         │
                                      Dropout 0.40
                                         │
                               Attention (Raffel 2015)
                                         │
                               Dense(1, sigmoid)  +  Adam + BCE
```

Single-modal embeddings use GloVe Twitter only. Multi-modal embeddings fall back to the mean of emoji2vec rows when a token is an emoji (or a cluster of them) and is missing from GloVe.

Classical models skip the sequence encoder: they average every in-vocab token into a 200-d vector, or concatenate that average with a 200-d emoji average (400-d).

Write-up: [docs/architecture.md](docs/architecture.md) and [docs/methodology.md](docs/methodology.md).

## Quick start (examples, no GloVe)

The example scripts only need Python 3.10+ and NumPy.

```bash
python3 -m pip install -r requirements-examples.txt

# Dataset census, split overlap, cue rates
python3 examples/inspect_dataset.py --root dataset --out docs/generated/dataset_report.md

# Tokenize a few tweets the same way the project thinks about tokens
python3 examples/tokenize_tweets.py --n 8

# Emoji ↔ label association (PMI, co-occurrence)
python3 examples/emoji_signals.py --root dataset

# Toy 8-d GloVe + emoji2vec average / concat
python3 examples/embedding_average.py

# NumPy clone of attention_layer.Attention
python3 examples/attention_demo.py

# Lexical logistic baseline on the real CSVs (stdlib + NumPy)
python3 examples/lexical_baseline.py --root dataset
```

```bash
python3 -m unittest discover -s tests -v
```

Narrative for each script: [examples/README.md](examples/README.md).

## Quick start (original notebooks)

1. Install [requirements.txt](requirements.txt).
2. Download [GloVe Twitter 27B 200-d](https://nlp.stanford.edu/projects/glove/) and convert it to word2vec binary, **or** point the notebooks at a text dump named `glove_tt.txt` (the metrics notebook used that name).
3. Keep `emoji2vec_twitter.bin` next to the notebooks (already in this repo).
4. Open `baseline_models.ipynb` or `evaluate_loaded_dl_models.ipynb`.

The Keras `SavedModel` directories under `model/` currently contain only `saved_model.pb` + `keras_metadata.pb` (no `variables/`). Reloading weights needs the missing variable shards. The Python model factory in `dl_model.py` is the source of truth for architecture.

## License

MIT. Original copyright line in [LICENSE](LICENSE) is from the 2023 course submission.
