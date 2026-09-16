# Multimodal Twitter sarcasm detection (emoji + text)

Personal final project for **UCPH Computational Cognitive Science 2 (2023)**.

The question is simple and still hard: can a model tell sarcastic tweets from sincere ones when the useful signal is split across **words**, **hashtags**, and **emoji**? This repo compares a single-modal text pipeline (Twitter GloVe) with a multimodal pipeline that also uses **emoji2vec**, first with classical classifiers and then with a bidirectional LSTM plus attention.

This is a personal academic archive. It is not a product, not a company codebase, and it does not call any social-media APIs.

## Why the problem is interesting

Sarcasm on Twitter is rarely a single inverted adjective. A tweet can look positive in isolation (`I just love having grungy ass hair`) and become sarcastic only after `#not` or `😑`. Other tweets use emoji as affect amplifiers rather than as flip signals (`😍` on a sincere reunion). A model that averages word vectors and ignores emoji therefore throws away a channel that humans use constantly.

Two design choices follow from that:

1. **Single-modal (`_w`)** — each tweet is represented from word embeddings only.
2. **Multimodal (`_we`)** — the same tweet is represented from word embeddings **plus** emoji2vec, either by concatenating averaged vectors (classical models) or by writing emoji vectors into the embedding matrix when a token is an emoji (deep models).

## What is in this repository

| Path | Role |
| --- | --- |
| `dataset/` | Train / test / emoji-heavy subtest splits (CSV) |
| `data_utils.py` | Tweet tokenization, averaged embeddings, Keras padding |
| `dl_model.py` | BiLSTM + attention architecture (`PrepModel`) |
| `attention_layer.py` | Temporal attention layer (Raffel et al., 2015) |
| `baseline_models.ipynb` | SVM, decision tree, random forest, gradient boosting |
| `evaluate_loaded_dl_models.ipynb` | Load saved Keras models and score them |
| `get_metrics_of_models.ipynb` | Accuracy / F1 / precision / recall comparison |
| `baseline_models/` | Saved sklearn pickles (partial; see docs) |
| `model/` | Saved Keras checkpoints (weights may be incomplete) |
| `emoji2vec.bin`, `emoji2vec_twitter.bin` | Pretrained emoji embeddings |
| `docs/` | Dataset, architecture, results, and reproduction notes |
| `examples/` | Stdlib-only walkthroughs that run without TensorFlow |

## Headline results (from the original notebooks)

Numbers below are copied from the executed notebook outputs. They are **not** re-measured in this documentation pass.

| Model | Test acc (text) | Test acc (text+emoji) | Subtest acc (text) | Subtest acc (text+emoji) |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| Gradient boosting | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| SVM | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| Random forest | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| BiLSTM + attention | 0.8635 | **0.8735** | 0.8669 | **0.8921** |

The multimodal deep model is the best of the set, and the emoji-heavy **subtest** is where emoji2vec helps most (random forest +4.7 points; BiLSTM +2.5 points). On the balanced 2,000-tweet test set the classical gain is small; SVM even drops slightly when emoji vectors are concatenated. That pattern is discussed in [`docs/models-and-results.md`](docs/models-and-results.md).

## Dataset snapshot

| Split | Tweets | Positive (sarcastic) rate | Typical length |
| --- | ---: | ---: | --- |
| `dataset/train_*.csv` | 39,780 | 46.5% | median 16 tokens |
| `dataset/test_*.csv` | 2,000 | 50.0% | median 16 tokens |
| `dataset/subtest_*.csv` | 278 | 61.9% | median 17 tokens |

The training file contains thousands of explicit sarcasm markers (`#not`, `#yeahright`, `#sarcastictweet`) and a long tail of emoji, led by 😂, 😊, 😭, 😍, ❤, and 😒. Mentions were already replaced with `<user>`. Full counts and caveats: [`docs/dataset.md`](docs/dataset.md).

## Quick start (examples, no heavy ML stack)

The scripts under `examples/` use only the Python standard library and the CSVs already in `dataset/`. They do **not** download tweets and do **not** need GloVe or TensorFlow.

```bash
python3 examples/inspect_dataset.py
python3 examples/tokenize_demo.py --limit 8
python3 examples/emoji_cooccurrence.py --top 15
python3 examples/heuristic_baseline.py
python3 examples/attention_walkthrough.py
```

Run the unit tests the same way:

```bash
python3 -m unittest discover -s tests -v
```

## Reproducing the original notebooks

That path needs extra files that are **not** all in git (Twitter GloVe 200d, some sklearn pickles, full Keras weight shards). See [`docs/reproducing-experiments.md`](docs/reproducing-experiments.md) before trying `baseline_models.ipynb`.

Suggested environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will also need NLTK's tweet tokenizer data the first time `data_utils.ReadOpen` runs (`nltk.download("punkt")` is not enough; install the Twitter-aware tokenizer resources your NLTK version expects).

## Documentation map

- [`docs/dataset.md`](docs/dataset.md) — splits, labels, hashtags, emoji, leakage risks
- [`docs/preprocessing.md`](docs/preprocessing.md) — `ReadOpen` → averaged vectors → Keras sequences
- [`docs/architecture.md`](docs/architecture.md) — classical vs BiLSTM+attention
- [`docs/attention.md`](docs/attention.md) — how `Attention` pools a sequence
- [`docs/models-and-results.md`](docs/models-and-results.md) — metric tables and reading of the results
- [`docs/reproducing-experiments.md`](docs/reproducing-experiments.md) — file checklist and notebook order
- [`docs/limitations.md`](docs/limitations.md) — what this 2023 project does not claim
- [`examples/README.md`](examples/README.md) — runnable walkthroughs

## License

MIT. See [`LICENSE`](LICENSE). Original upload copyright is `pang990801` (2023).
