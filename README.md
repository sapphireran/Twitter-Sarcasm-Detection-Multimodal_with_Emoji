# Multimodal Twitter Sarcasm Detection with Emoji

Personal 2023 University of Copenhagen project for **Computational Cognitive Science 2**.
The question the repo answers is narrow: *does adding emoji embeddings to tweet text help a sarcasm classifier, and where does that help show up?*

This is a personal academic archive (MIT, 2023). It is not a product, and it does not contain company code.

## What is in here

Two modeling tracks share one tweet corpus:

| Track | Input | Model family | Best test accuracy in the original notebooks |
| --- | --- | --- | --- |
| Single-modal | GloVe-Twitter 200d word vectors | SVM / trees / BiLSTM+attention | **86.35%** (BiLSTM+Attn, word only) |
| Multi-modal | word vectors **plus** emoji2vec 200d | same families, concatenated or mixed embeddings | **87.35%** (BiLSTM+Attn, word+emoji) |

The emoji signal is small on the balanced 2,000-tweet test set and much clearer on the **emoji-rich subtest** (278 tweets, 266 of which contain emoji): multi-modal BiLSTM+attention reaches **89.21%** accuracy and **0.911** F1 there.

Recorded notebook numbers, split sizes, and cue statistics are written out in [`docs/`](docs/README.md). Runnable, dependency-light walkthroughs live in [`examples/`](examples/README.md).

## Repository map

```
attention_layer.py      Raffel-style temporal attention (Keras Layer)
dl_model.py             stacked BiLSTM + Attention + sigmoid head
data_utils.py           tweet tokenization, GloVe / emoji2vec pooling
dataset/                train / test / subtest sentence+label CSVs
baseline_models/        pickled sklearn baselines from the 2023 run
baseline_models.ipynb   train / load SVM, DT, RF, GBT
evaluate_loaded_dl_models.ipynb
get_metrics_of_models.ipynb
model/                  saved Keras single- and multi-modal weights
emoji2vec*.bin          emoji embedding tables used in 2023
docs/                   project notes written from the files above
examples/               scripts that do not need GloVe or TensorFlow
tests/                  checks for the docs examples and CSV splits
```

GloVe-Twitter `27B.200d` is **not** stored in git (it is large). `emoji2vec_twitter.bin` is. How to rebuild the 2023 environment is in [`docs/reproducing.md`](docs/reproducing.md).

## Dataset snapshot

Counts were recomputed from the CSVs in this clone:

| Split | Tweets | Sarcastic (`1`) | Non-sarcastic (`0`) | Emoji tweets | Explicit sarcasm-cue hashtags |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 39,780 | 18,488 (46.5%) | 21,292 | 5,223 | 3,480 |
| `test` | 2,000 | 1,000 (50%) | 1,000 | 266 | 595 |
| `subtest` | 278 | 172 (61.9%) | 106 | 266 | 129 |

Labels are binary. A tweet is one line in `*_sentence.csv`; the aligned label is the same line number in `*_label.csv`. Tokenization in the original code uses NLTK `TweetTokenizer` and lowercasing. See [`docs/dataset.md`](docs/dataset.md).

Typical sarcastic cues in this collection are hashtags such as `#not`, `#sarcasm`, `#sarcastictweet`, and `#yeahright`, often paired with an emoji that undercuts a positive surface sentence:

```text
I loovee when people text back ... 😒 #sarcastictweet
feeling like a million bucks after that chem 2 test . 😅 #not
```

## Model sketch

`PrepModel` in `dl_model.py` is:

1. Frozen 200-d embedding table (GloVe words; emoji tokens can be filled from emoji2vec).
2. Dropout 0.25.
3. Bidirectional LSTM 256 (sequences on).
4. Dropout 0.4.
5. Second bidirectional LSTM 256 (sequences on).
6. Dropout 0.4.
7. Raffel feed-forward attention (`attention_layer.py`).
8. Dense(1, sigmoid), Adam `lr=0.001`, binary cross-entropy.

Classical baselines average the same 200-d (or 400-d concatenated) vectors per tweet and fit SVM, decision tree, random forest, or gradient boosting. Details and a layer-by-layer parameter count are in [`docs/architecture.md`](docs/architecture.md).

## Results that motivated the write-up

From `get_metrics_of_models.ipynb` / `evaluate_loaded_dl_models.ipynb` (rounded):

| Model | Test acc word | Test acc word+emoji | Subtest acc word | Subtest acc word+emoji |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.769 | 0.763 | 0.813 | 0.824 |
| Decision tree | 0.727 | 0.730 | 0.777 | 0.799 |
| Random forest | 0.815 | 0.818 | 0.806 | **0.853** |
| Gradient boosting | 0.746 | 0.748 | 0.795 | 0.795 |
| BiLSTM + attention | 0.864 | **0.874** | 0.867 | **0.892** |

Emoji features barely move the full test set for linear / tree models and even slightly *hurt* SVM. They help more when evaluation is restricted to tweets that actually contain emoji. The stacked recurrent model is well ahead of the pooled-vector baselines on both splits. Interpretation notes are in [`docs/experiments.md`](docs/experiments.md).

## Personal examples (no GloVe, no TensorFlow)

The `examples/` scripts only need the Python standard library plus NumPy. They walk through the same ideas as `data_utils.py` without downloading 200-d Twitter GloVe:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/01_dataset_overview.py
python3 examples/02_tokenize_tweets.py
python3 examples/03_sarcasm_cues.py
python3 examples/04_tiny_embedding_pipeline.py
python3 examples/05_attention_math.py
python3 examples/06_toy_baseline.py
python3 examples/07_split_consistency.py
python3 examples/08_format_results_table.py
```

```bash
python3 -m unittest discover -s tests -v
```

## Original notebooks

| Notebook | What it did in 2023 |
| --- | --- |
| `baseline_models.ipynb` | Load GloVe + emoji2vec, pool tweets, fit or reload sklearn models |
| `evaluate_loaded_dl_models.ipynb` | Reload the two Keras `SavedModel` directories and score test / subtest |
| `get_metrics_of_models.ipynb` | Accuracy, F1, precision, recall, and comparison plots |

Paths inside the notebooks are inconsistent (`train_sentence.csv` vs `dataset/train_sentence.csv`, `glove.twitter.27B.200d.bin` vs `glove_tt.txt`). [`docs/reproducing.md`](docs/reproducing.md) lists the files you actually need and the path fixes.

## License

MIT License. Copyright (c) 2023 pang990801. See [LICENSE](LICENSE).
