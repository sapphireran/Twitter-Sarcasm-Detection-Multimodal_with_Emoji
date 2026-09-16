# Multimodal Twitter sarcasm detection (text + emoji)

Personal 2023 course project for **Computational Cognitive Science 2** at the University of Copenhagen. The question is whether emoji embeddings add anything useful on top of tweet text when the task is binary sarcasm detection.

The original notebooks train and score:

- classical baselines (SVM, decision tree, random forest, gradient boosting) on averaged word vectors
- a bidirectional LSTM with a Raffel-style attention layer, once with GloVe-only embeddings and once with emoji2vec fallbacks for emoji tokens

This repository now also has a **docs/** and **examples/** tree that you can run without GloVe, Keras, or the unpublished `glove.twitter.27B.200d.bin` file.

## What is in the repo

| Path | Role |
| --- | --- |
| `dataset/` | Train / test / emoji-rich subtest tweets and labels |
| `data_utils.py` | Tweet loading, averaged vectors, Keras-style padding + embedding matrix |
| `dl_model.py` | BiLSTM + attention classifier |
| `attention_layer.py` | Temporal attention (Raffel et al., 2015) |
| `baseline_models/` | Pickled sklearn baselines (decision tree and gradient boosting are present) |
| `model/` | Saved Keras single-modal and multimodal checkpoints |
| `emoji2vec.bin`, `emoji2vec_twitter.bin` | Pretrained emoji vectors |
| `docs/` | Dataset card, method, recorded metrics, reproduction notes |
| `examples/` | Runnable inspection, emoji-signal, lexical-baseline, and attention demos |

GloVe Twitter 200-d vectors are **not** checked in. The notebooks expect `glove.twitter.27B.200d.bin` (or `glove_tt.txt` in `get_metrics_of_models.ipynb`) next to the project root.

## Labels and splits

`1` is the sarcastic class. That reading is consistent with the hashtag supervision in the files: `#not`, `#sarcasm`, `#yeahright`, and `#sarcastictweet` sit almost entirely in class `1`.

| Split | Tweets | Class 0 | Class 1 | Notes |
| --- | --- | --- | --- | --- |
| `train` | 39,780 | 21,292 (53.5%) | 18,488 (46.5%) | Mixed text; ~13.8% contain emoji |
| `test` | 2,000 | 1,000 | 1,000 | Balanced; no `<user>` tokens |
| `subtest` | 278 | 106 | 172 | Almost every tweet has emoji (277 / 278) |

Sentence files are line-oriented, not columnar CSV. Some lines are wrapped in quotes; some contain commas. The original loader (`ReadOpen`) replaces commas with spaces. See [docs/dataset.md](docs/dataset.md).

## Recorded test-set scores

Numbers below are copied from the executed `get_metrics_of_models.ipynb` / `baseline_models.ipynb` outputs. `W` is text-only (GloVe). `WE` concatenates or falls back to emoji2vec.

| Model | Test acc (W) | Test acc (WE) | Subtest acc (W) | Subtest acc (WE) |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.727 | 0.730 | 0.777 | 0.799 |
| SVM | 0.769 | 0.763 | 0.813 | 0.824 |
| Gradient boosting | 0.746 | 0.748 | 0.795 | 0.795 |
| Random forest | 0.815 | 0.818 | 0.806 | 0.853 |
| BiLSTM + attention | **0.864** | **0.874** | 0.867 | **0.892** |

Emoji vectors help most on the emoji-rich subtest, and they help the sequence model more than they help bag-of-vectors SVM. Full precision / recall / F1 tables are in [docs/results.md](docs/results.md).

## Architecture (deep model)

```
tokens
  → frozen 200-d embedding (GloVe, with optional emoji2vec for emoji OOV)
  → Dropout(0.25)
  → BiLSTM(256)          # 512-d after both directions
  → Dropout(0.4)
  → BiLSTM(256)
  → Dropout(0.4)
  → Attention            # tanh scoring, masked softmax, weighted sum
  → Dense(1, sigmoid)
```

Optimizer: Adam (`lr=0.001`). Loss: binary cross-entropy. The attention layer follows [Raffel et al., 2015](https://arxiv.org/abs/1512.08756) and is documented in [docs/methodology.md](docs/methodology.md).

## Lightweight examples (no GloVe / Keras)

```bash
python3 examples/inspect_dataset.py
python3 examples/emoji_signals.py
python3 examples/lexical_baseline.py
python3 examples/attention_demo.py
python3 examples/preprocess_walkthrough.py
```

`lexical_baseline.py` trains a from-scratch multinomial Naive Bayes model on tweet tokens plus hashtag / emoji indicators. It is a documentation baseline, not a replacement for the Keras model.

```bash
python3 -m unittest discover -s tests -v
```

## Original notebooks

1. `baseline_models.ipynb` — train or load sklearn models; print accuracy.
2. `evaluate_loaded_dl_models.ipynb` — reload the two Keras checkpoints.
3. `get_metrics_of_models.ipynb` — accuracy, F1, precision, recall, and the comparison plot.

Reproduction caveats (missing GloVe file, Gensim 4 API break, notebook copy-paste in the “train if missing” branches) are listed in [docs/reproduction.md](docs/reproduction.md).

## License

MIT. The license header names `pang990801` (2023).
