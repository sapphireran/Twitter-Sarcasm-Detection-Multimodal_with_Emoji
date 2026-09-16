# Multi-modal sarcasm detection with emoji

Personal 2023 final project for **Computational Cognitive Science 2** at the University of Copenhagen. The question: if you already have Twitter GloVe for the words, does adding **emoji2vec** help you detect sarcasm — especially on tweets where wording and emoji co-occur?

This snapshot keeps the original notebooks, the frozen Bi-LSTM checkpoints, two emoji2vec binaries, and the train/test/subtest CSVs. The `docs/` and `examples/` trees are a later write-up of that work so the repo can be read without reopening Jupyter.

**Scope:** personal course code and notes only. Nothing here is company or work product.

## Result in one table

From the executed 2023 notebooks (`get_metrics_of_models.ipynb`). Positive class = sarcastic.

| Model | `test` text | `test` +emoji | `subtest` text | `subtest` +emoji |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.769 | 0.763 | 0.813 | 0.824 |
| Random forest | 0.814 | 0.818 | 0.806 | 0.853 |
| Bi-LSTM + Attention | 0.8635 | **0.8735** | 0.867 | **0.892** |

`subtest` is 278 tweets that almost all contain an emoji. That is where emoji2vec is supposed to matter, and it does: RF +4.7 acc, Bi-LSTM +2.5. On the mixed test set the deep multi-modal model is mostly a **precision** gain (0.853 → 0.904) with a recall drop. Full P/R/F1 and caveats: [docs/evaluation.md](docs/evaluation.md).

A hashtag-only rule on the current CSVs already hits **0.806 acc / 1.00 precision** on `test` because every `#not` / `#sarcasm` there sits on a positive label. Beat that before claiming a new architecture understood irony.

## How the model looks

```text
tokens ─┬─ mean GloVe (200)  (+ mean emoji2vec → 400-d)  → SVM / trees / RF / GBT
        └─ 200-d sequence (GloVe or emoji2vec per token)
               Dropout → Bi-LSTM 256 → Bi-LSTM 256 → Attention → sigmoid
```

- Single-modal deep model: emoji / OOV rows are zeros.
- Multi-modal deep model: those rows are the mean emoji2vec vector, **same 200-d width**. Trainable parameter count does not change.
- sklearn multi-modal models concatenate to 400-d instead.

Details: [docs/architecture.md](docs/architecture.md), [docs/embeddings.md](docs/embeddings.md), [docs/preprocessing.md](docs/preprocessing.md).

## Repository layout

```text
dataset/                  train / test / subtest sentence+label pairs
model/best_model_*        2023 Keras SavedModels (single- and multi-modal)
baseline_models/          some sklearn pickles (DT, GBT; SVM/RF missing)
emoji2vec_twitter.bin     200-d table the notebooks load (1,661 emoji)
emoji2vec.bin             300-d upstream-style dump (same keys, other space)
data_utils.py             tokenize, mean-pool, build the embedding matrix
dl_model.py               PrepModel: frozen embed + 2× Bi-LSTM + Attention
attention_layer.py        Raffel-style tanh attention (Keras layer)
*.ipynb                   original train / eval / metrics notebooks
docs/                     written-up dataset, train, eval, references
examples/                 runnable NumPy / stdlib scripts
```

GloVe Twitter 27B 200-d is **not** in the repo. You need it to retrain or to rebuild `X` for the SavedModels. Emoji geometry can be inspected without it.

## Examples (no TensorFlow, no GloVe)

```bash
python examples/explore_dataset.py          # split sizes, cue leakage, histograms
python examples/tokenize_tweets.py          # comma collapse + tweet-ish tokens
python examples/inspect_emoji2vec.py --compare
python examples/attention_demo.py           # NumPy clone of attention_layer.py
python examples/cue_baseline.py             # live hashtag / clash scores
python examples/reprint_course_results.py   # 2023 table, historical
```

Needs Python 3.10+ and NumPy. See [examples/README.md](examples/README.md).

## Retrain / reload (full stack)

```bash
pip install -r requirements.txt
# download glove.twitter.27B.200d.txt and convert; see docs/embeddings.md
```

Then either open the notebooks (path caveats in [docs/notebooks.md](docs/notebooks.md)) or follow the fit recipe in [docs/training.md](docs/training.md). Modern TensorFlow wants `custom_objects={"Attention": Attention}` and `Adam(learning_rate=...)`.

## Documentation index

| Page | Contents |
| --- | --- |
| [docs/dataset.md](docs/dataset.md) | Splits, cue hashtags, mention/emoji shifts |
| [docs/preprocessing.md](docs/preprocessing.md) | `ReadOpen`, mean-pool vs Keras pad |
| [docs/embeddings.md](docs/embeddings.md) | GloVe Twitter + both emoji2vec files |
| [docs/architecture.md](docs/architecture.md) | Baselines and Bi-LSTM + Attention |
| [docs/training.md](docs/training.md) | How the 2023 models were fit |
| [docs/evaluation.md](docs/evaluation.md) | Metrics, gains, reload notes |
| [docs/notebooks.md](docs/notebooks.md) | Cell-by-cell map of the three `.ipynb`s |
| [docs/references.md](docs/references.md) | Papers and tooling |

## License

MIT. Original copyright line is in `LICENSE` (2023 `pang990801`).
