# Twitter sarcasm detection with emoji (personal course project)

Personal University of Copenhagen **Computational Cognitive Science 2**
(2023) final project: *Multi-modal Sarcasm Detection Using Textual
Contents and Emoji Co-occurrences in Twitter*.

Author: Sapphire Ran. License: MIT. This repository is a personal
academic archive. It is not company code and is not affiliated with any
employer.

The original 2023 work trains classical baselines and a bidirectional
LSTM with temporal attention, with and without emoji2vec. This branch
adds a documentation set and runnable examples so the project can be
read without the missing GloVe table or TensorFlow.

## Results at a glance

Recorded June 2023. Full tables and caveats:
[docs/experiments.md](docs/experiments.md).

| Model | Test acc (word) | Test acc (word+emoji) | Emoji-subtest acc (word) | Emoji-subtest acc (word+emoji) |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 72.65 | 72.95 | 77.70 | 79.86 |
| SVM | 76.90 | 76.30 | 81.29 | 82.37 |
| Gradient boosting | 74.60 | 74.75 | 79.50 | 79.50 |
| Random forest | 81.45 | 81.80 | 80.58 | 85.25 |
| Bi-LSTM + attention | **86.35** | **87.35** | **86.69** | **89.21** |

The emoji-only subtest is 278 tweets filtered from the 2,000-tweet test
set. That is where the second modality actually exists.

## Repository layout

```
dataset/                 train / test / subtest sentence+label CSVs
data_utils.py            2023 readers, mean pooling, Keras preprocess
dl_model.py              Bi-LSTM + attention constructor
attention_layer.py       Raffel-style Keras attention
baseline_models.ipynb    sklearn baselines
evaluate_loaded_dl_models.ipynb
get_metrics_of_models.ipynb
docs/                    project notes written after the course
examples/                scripts that run with NumPy only
tests/                   unittest coverage for the example library
```

## Run the personal examples

From the repository root, with NumPy on `PYTHONPATH` (the system
interpreter on this machine already has it):

```bash
python3 -m examples.inspect_dataset
python3 -m examples.reported_results
python3 -m examples.tokenize_tweets --split subtest --limit 8
python3 -m examples.attention_walkthrough
python3 -m examples.multimodal_fusion
python3 -m examples.embedding_matrix_walkthrough
python3 -m examples.lexical_baseline
python3 -m unittest discover -s tests -v
```

See [examples/README.md](examples/README.md) for flags and what each
script prints.

## Reproduce the 2023 notebooks

You need GloVe Twitter 27B 200-d, gensim, NLTK, the `emoji` package,
scikit-learn, and a TensorFlow that still accepts the 2023 SavedModel
plus `Adam(lr=...)`. The large GloVe file and the Keras weight shards
are **not** in git.

Step-by-step constraints, path mismatches, and gensim 4 notes:
[docs/reproduction.md](docs/reproduction.md).

`requirements.txt` lists the original import surface.

## Documentation

- [docs/project-overview.md](docs/project-overview.md) — question and design
- [docs/data-pipeline.md](docs/data-pipeline.md) — splits and features
- [docs/architecture.md](docs/architecture.md) — LSTM and attention math
- [docs/experiments.md](docs/experiments.md) — metric cards
- [docs/notebook-map.md](docs/notebook-map.md) — which notebook to open
- [docs/glossary.md](docs/glossary.md) — project-specific terms

## Data

39,780 train tweets (46.5% sarcastic), 2,000 balanced test tweets, and
a 278-tweet emoji subset of test. Labels are `0` (literal) / `1`
(sarcastic). Test and subtest are stored as one sarcastic block then one
literal block; train is mostly grouped but not a single clean partition.
Shuffle before training. Details and cue-rate tables are in the
data-pipeline note.
