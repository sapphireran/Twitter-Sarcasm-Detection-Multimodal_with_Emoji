# Twitter sarcasm detection with text and emoji

Personal final project for **UCPH Computational Cognitive Science 2 (2023)**.

The question I wanted an answer to: if a tweet is sarcastic, does the emoji stream actually help a classifier, or is it mostly decorative noise on top of the words?

Short version of what I found in June 2023:

- A frozen GloVe embedding + stacked BiLSTM + attention already does most of the work (86.4% test accuracy from text alone).
- Adding emoji2vec into the same embedding table adds about **+1.0 point** on the balanced test set and **+2.5 points** on the emoji-heavy subtest.
- Classical baselines sit well below that. Random forest is the least embarrassing of them, especially on the emoji slice.

This repo is the personal lab dump: preprocessing, the Keras model, saved 2023 metrics, and the notes I wish I had written while the runs were still warm.

## What "multi-modal" means here

There is no image stream. Both channels are still text-adjacent:

| Channel | Source | How it enters the model |
| --- | --- | --- |
| Words | GloVe Twitter 27B, 200-d | token embedding, frozen |
| Emoji | emoji2vec (Twitter-trained 200-d) | same table, used when GloVe misses a token that is actually an emoji |

Classical models get a cheaper version of the same idea: mean-pool GloVe over the tweet for the single-modal run, then concatenate a mean-pooled emoji2vec vector for the multi-modal run.

## Data

All three splits live in `dataset/`. Labels are `1` = sarcastic, `0` = not.

| Split | Rows | Sarcastic | Emoji-bearing tweets | Notes |
| --- | --- | --- | --- | --- |
| `train_*` | 39,780 | 46.5% | 13.8% | slightly more non-sarcastic |
| `test_*` | 2,000 | 50.0% | 13.9% | the number I quote as "full test" |
| `subtest_*` | 278 | 61.9% | 99.6% | almost every row has at least one emoji |

The subtest is not a random 278-row draw. It is the emoji-rich slice. That is why multi-modal gains look larger there: the extra channel is actually present.

Tweets were already lightly social-media-normalized (`<user>` mentions, hashtags left intact). Mean length is about 16–18 whitespace tokens. I tokenize again with NLTK `TweetTokenizer` before anything else.

I did not redistribute GloVe. `glove.twitter.27B.200d.bin` (or the text form `glove_tt.txt` I used in one notebook) has to be downloaded separately.

## Layout

```
attention_layer.py              # Raffel-style temporal attention
data_utils.py                   # read / tokenize / embed
dl_model.py                     # BiLSTM + attention constructor
baseline_models.ipynb           # SVM / DT / RF / GBT train-or-load
evaluate_loaded_dl_models.ipynb # reload the two Keras snapshots
get_metrics_of_models.ipynb     # accuracy / F1 / plots from June 2023
dataset/                        # train, test, subtest
baseline_models/                # some sklearn pickles (incomplete)
model/                          # SavedModel graphs (weights folder missing)
notes/                          # personal experiment notes
emoji2vec.bin
emoji2vec_twitter.bin           # the one the notebooks actually load
```

## Models

**Deep model (`dl_model.PrepModel`)**

1. Frozen 200-d embedding table (`count` × 200).
2. Dropout 0.25.
3. Bidirectional LSTM 256 (so 512-d), `return_sequences=True`.
4. Dropout 0.4.
5. Another BiLSTM 256, sequences kept.
6. Dropout 0.4.
7. Attention over time (Raffel et al., [arXiv:1512.08756](https://arxiv.org/abs/1512.08756)).
8. Dense sigmoid, Adam, binary cross-entropy.

Single-modal vs multi-modal is not two architectures. It is two ways of filling the embedding table in `data_utils.Preprocess`. When `get_emoji2vec=False`, unknown emoji tokens stay at zero. When it is `True`, they get the mean emoji2vec vector.

**Baselines**

Default sklearn SVM, decision tree, random forest, gradient boosting. Input is a 200-d mean GloVe vector, or a 400-d GloVe+emoji2vec concat.

There are copy-paste mistakes in a few "train if missing" cells (a decision-tree / forest / boosting branch that accidentally fits `SVC()`). The numbers I trust are the ones already printed in `get_metrics_of_models.ipynb`, from pickles that existed in June 2023.

## Headline numbers (June 2023)

Rounded from the executed notebook. `W` = words only. `WE` = words + emoji.

**Full test (2,000 rows)**

| Model | Acc W | Acc WE | F1 W | F1 WE |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.727 | 0.730 | 0.756 | 0.757 |
| SVM | 0.769 | 0.763 | 0.772 | 0.766 |
| Gradient boosting | 0.746 | 0.748 | 0.751 | 0.753 |
| Random forest | 0.815 | 0.818 | 0.823 | 0.826 |
| BiLSTM + attention | **0.864** | **0.874** | 0.866 | 0.869 |

**Emoji subtest (278 rows)**

| Model | Acc W | Acc WE | F1 W | F1 WE |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.777 | 0.799 | 0.834 | 0.846 |
| SVM | 0.813 | 0.824 | 0.852 | 0.853 |
| Gradient boosting | 0.795 | 0.795 | 0.838 | 0.836 |
| Random forest | 0.806 | 0.853 | 0.852 | 0.884 |
| BiLSTM + attention | **0.867** | **0.892** | 0.894 | 0.911 |

Emoji helps most where emoji actually appear. SVM even got slightly *worse* on the full test when I concatenated the emoji mean. That is one of the reasons I stopped treating "more modalities" as automatically better.

Precision / recall and the original bar plots are in `notes/2023-06-experiment-log.md` and `get_metrics_of_models.ipynb`.

## How I ran this in 2023

Environment was a local TensorFlow 2 / Keras setup with `gensim` KeyedVectors, `nltk.TweetTokenizer`, and the `emoji` package.

Typical deep-model path:

1. Load GloVe Twitter 200-d and `emoji2vec_twitter.bin`.
2. `ReadOpen` on the three CSV pairs.
3. `Preprocess(..., get_emoji2vec=True/False)` on train; `preprocess_test` on test / subtest so they share the train tokenizer and pad length (78 in the saved graphs).
4. `PrepModel` and `model.fit`. I kept the better snapshot of each setting.

Typical baseline path: `ml_read_data` → sklearn `fit` → `joblib.dump` under `baseline_models/`.

`evaluate_loaded_dl_models.ipynb` only reloads `model/best_model_{single,multi}_modal`. Those folders currently have the SavedModel proto and `keras_metadata.pb` but not the `variables/` shard, so they will not restore on a clean machine.

## Personal notes

I wrote these after the fact from the notebooks, printed metrics, and a fresh pass over the CSVs. They are lab notes, not a paper.

- [notes/2023-06-experiment-log.md](notes/2023-06-experiment-log.md) — what I ran, the tables, what I think they mean
- [notes/dataset-notes.md](notes/dataset-notes.md) — split construction, emoji rate, examples
- [notes/architecture-and-training.md](notes/architecture-and-training.md) — why this stack, attention, embedding tricks
- [notes/reproduction-checklist.md](notes/reproduction-checklist.md) — files that are here vs files I still need locally
- [notes/open-questions.md](notes/open-questions.md) — what I still would not claim

## License

MIT. Copyright (c) 2023 pang990801 / Sapphire Ran.
