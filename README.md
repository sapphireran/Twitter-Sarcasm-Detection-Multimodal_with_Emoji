# Multimodal Twitter sarcasm detection with emoji co-occurrence

Personal 2023 final project for **Computational Cognitive Science 2** at the University of Copenhagen. The question is whether emoji embeddings add anything useful on top of tweet text when the label is sarcasm.

Short version: they do, but mostly on the emoji-heavy slice. A stacked bidirectional LSTM with attention is the best model in the saved notebooks. Random forest is the strongest classical baseline.

This repository now also has a self-contained `docs/` and `examples/` tree so the pipeline can be read and exercised without downloading GloVe or TensorFlow.

## Why this problem

Sarcasm is often an incongruity: the words say one thing and the delivery says another. On Twitter that delivery is frequently an emoji, a hashtag (`#not`, `#sarcastictweet`), or both.

```
I just love getting shots 💉 #sarcastictweet     → sarcastic
Rest in peace & love to you and your family      → not sarcastic
```

A bag of GloVe averages can see the positive words `love` / `getting` / `shots`. It cannot see that `💉` plus `#sarcastictweet` flips the intent. The multimodal setup therefore keeps a second 200-dimensional channel from [emoji2vec](https://github.com/uclnlp/emoji2vec) and either concatenates it (classical models) or writes those vectors into the same embedding matrix (deep model).

## Results at a glance

Numbers are copied from the executed cells in `get_metrics_of_models.ipynb`. **Single** = GloVe tweet averages only. **Multi** = GloVe concatenated with emoji2vec (baselines) or a mixed embedding table (Bi-LSTM). The **subtest** is 278 tweets that are 99.3% emoji-bearing and heavier on `#not` / `#sarcastictweet`.

| Model | Test acc (single / multi) | Subtest acc (single / multi) | Test F1 (single / multi) | Subtest F1 (single / multi) |
| --- | --- | --- | --- | --- |
| SVM | 0.769 / 0.763 | 0.813 / 0.824 | 0.772 / 0.766 | 0.852 / 0.853 |
| Decision tree | 0.727 / 0.730 | 0.777 / 0.799 | 0.756 / 0.757 | 0.834 / 0.846 |
| Random forest | 0.815 / 0.818 | 0.806 / 0.853 | 0.823 / 0.826 | 0.852 / 0.884 |
| Gradient boosting | 0.746 / 0.748 | 0.795 / 0.795 | 0.751 / 0.753 | 0.838 / 0.836 |
| Bi-LSTM + attention | **0.864 / 0.874** | **0.867 / 0.892** | **0.866 / 0.869** | **0.894 / 0.911** |

The multimodal lift is small on the balanced 2,000-tweet test set and larger on the subtest, which is the slice where emoji should matter. Full tables, precision/recall, and caveats live in [docs/results.md](docs/results.md).

## Repository layout

```
.
├── attention_layer.py          Raffel-style temporal attention (Keras)
├── data_utils.py               Tokenize, average embeddings, pad sequences
├── dl_model.py                 Frozen-embedding Bi-LSTM stack + attention
├── baseline_models.ipynb       Train / load SVM, DT, RF, GBT
├── evaluate_loaded_dl_models.ipynb
├── get_metrics_of_models.ipynb Accuracy / F1 / precision / recall + plots
├── dataset/                    train / test / subtest sentences + labels
├── model/                      Saved single- and multi-modal Keras graphs
├── baseline_models/            Saved sklearn pickles (partial)
├── docs/                       Methodology, API, reproduction notes
└── examples/                   Runnable toy pipeline (numpy only)
```

## Data

| Split | Rows | Sarcastic (1) | Literal (0) | Notes |
| --- | --- | --- | --- | --- |
| `dataset/train_sentence.csv` | 39,780 | 18,488 (46.5%) | 21,292 (53.5%) | Main training pool |
| `dataset/test_sentence.csv` | 2,000 | 1,000 (50%) | 1,000 (50%) | Held-out evaluation |
| `dataset/subtest_sentence.csv` | 278 | 172 (61.9%) | 106 (38.1%) | Emoji / hashtag-heavy slice |

Each sentence file is one tweet per line. The matching `*_label.csv` is a single column of `0` / `1` with no header. User mentions were already replaced with the token `<user>`.

Inspect the real CSVs without any ML stack:

```bash
python3 examples/inspect_dataset.py
```

## Two model families

**Classical baselines** (`data_utils.ml_read_data` → sklearn):

1. Tweet-tokenize and lowercase.
2. Average every in-vocabulary GloVe vector → 200-d tweet embedding.
3. Average every in-vocabulary emoji2vec vector → 200-d emoji embedding (zeros if the tweet has no emoji).
4. Single-modal features = step 2. Multi-modal features = concatenate step 2 and step 3 → 400-d.
5. Fit SVM / decision tree / random forest / gradient boosting.

**Deep model** (`data_utils.Preprocess` → `dl_model.PrepModel`):

1. Keras `Tokenizer` + post-padding.
2. Embedding matrix of shape `(vocab, 200)`. Words copy GloVe. Emoji tokens copy the mean emoji2vec vector (or zeros in the single-modal ablation).
3. Frozen embedding → dropout 0.25 → Bi-LSTM 256 → dropout 0.4 → Bi-LSTM 256 → dropout 0.4 → [attention](attention_layer.py) → sigmoid.

Architecture notes and layer counts: [docs/models.md](docs/models.md).

## Run the lightweight examples

The original notebooks need GloVe Twitter 200-d, `emoji2vec_twitter.bin`, TensorFlow, and NLTK. The `examples/` package does **not**. It reimplements the same averaging, fusion, and attention math with deterministic hash embeddings so anyone can step through the idea on a laptop.

```bash
# no extra packages beyond numpy (already used by the project)
python3 examples/run_pipeline.py
python3 examples/run_attention.py
python3 examples/run_toy_classifier.py
python3 examples/inspect_dataset.py

# or everything plus the unit checks
python3 -m examples.run_all
python3 -m unittest discover -s tests -v
```

Walkthrough of what each script prints: [examples/README.md](examples/README.md).

## Reproduce the original notebooks

That path still needs the large embedding files and the 2023 Python / TensorFlow combo. See [docs/reproduction.md](docs/reproduction.md).

Minimum extra files the notebooks expect in the repo root:

- `glove.twitter.27B.200d.bin` (or `glove_tt.txt` in `get_metrics_of_models.ipynb`)
- `emoji2vec_twitter.bin` (already in this repo)

Then open `baseline_models.ipynb` and `evaluate_loaded_dl_models.ipynb`. Paths inside some cells assume the CSV files sit next to the notebook rather than under `dataset/`; the reproduction doc lists the mismatches.

## Documentation map

| Doc | Contents |
| --- | --- |
| [docs/methodology.md](docs/methodology.md) | Task definition, incongruity, why two channels |
| [docs/data-pipeline.md](docs/data-pipeline.md) | Tokenization, averaging, padding, split statistics |
| [docs/emoji-fusion.md](docs/emoji-fusion.md) | Early fusion vs mixed embedding table |
| [docs/models.md](docs/models.md) | Baselines, Bi-LSTM + attention, hyper-parameters |
| [docs/results.md](docs/results.md) | Full metric tables and how to read the lift |
| [docs/api-reference.md](docs/api-reference.md) | Functions in `data_utils.py`, `dl_model.py`, `attention_layer.py` |
| [docs/notebooks.md](docs/notebooks.md) | What each notebook actually does |
| [docs/reproduction.md](docs/reproduction.md) | Environment, files, known path bugs |
| [docs/known-issues.md](docs/known-issues.md) | Fallback-training bugs, Keras imports, missing GloVe |

## License

MIT. Original copyright notice is in [LICENSE](LICENSE).
