# Code map

A reading order for the original course files plus the 2026 personal
docs / examples. Paths are from the repository root.

## Start here

| File | What you learn |
| --- | --- |
| [README.md](../README.md) | Claim, splits, how to run the lightweight examples |
| [docs/dataset.md](dataset.md) | Whether you can trust the CSVs |
| [examples/explore_dataset.py](../examples/explore_dataset.py) | The same facts, generated from disk |

## Original training stack (2023)

Read these in order if you want to follow a tweet into a model.

1. **`data_utils.py`**
   * `ReadOpen` — line → lower-cased TweetTokenizer tokens
   * `AverageVectorPerTweet` / `AverageVectorPerEmoji` — mean pool
   * `ml_read_data` — W and WE matrices + a shared shuffle
   * `Preprocess` / `preprocess_test` — Keras ids, pad, embedding matrix
2. **`attention_layer.py`**
   * Keras 2 `Attention` layer, Raffel-style scores, mask + ε
3. **`dl_model.py`**
   * `PrepModel` — frozen embedding, two BiLSTMs, attention, sigmoid
4. **`baseline_models.ipynb`**
   * loads GloVe + emoji2vec, fits or reloads SVM / DT / RF / GBT
5. **`get_metrics_of_models.ipynb`**
   * accuracy / F1 / precision / recall for every W/WE × split cell
   * last cells export `metrics_data.csv` (not in the snapshot)
6. **`evaluate_loaded_dl_models.ipynb`**
   * shorter reload of `model/best_model_{single,multi}_modal`

Companion write-ups: [preprocessing.md](preprocessing.md),
[architecture.md](architecture.md).

## Personal example stack (2026)

No TensorFlow. NumPy + stdlib.

```
examples/lib/dataset.py       load_split, alignment check, summaries
examples/lib/tokenize.py      tweet-ish tokens, emoji / hashtag extractors
examples/lib/cues.py          #not rule + contrast pattern
examples/lib/attention.py     NumPy Attention.call
examples/lib/embeddings.py    dummy KeyedVectors, mean pool, pad
examples/lib/metrics.py       recorded June 2023 tables + metric_bundle
```

CLIs (thin wrappers, all accept `--help`):

```
examples/explore_dataset.py
examples/tokenize_tweets.py
examples/sarcasm_cues.py
examples/attention_numpy.py
examples/embedding_pipeline.py
examples/report_results.py
```

Tests: `python3 -m unittest tests.test_examples`.

## Artefacts you should not expect to "just run"

| Path | State in this snapshot |
| --- | --- |
| `emoji2vec_twitter.bin` | present; needs gensim 3 to load |
| `emoji2vec.bin` | present; unused by the notebooks |
| `glove.twitter.27B.200d.bin` | **absent** |
| `baseline_models/dt_*.pkl`, `gbt_*.pkl` | present |
| `baseline_models/svm_*.pkl`, `rf_*.pkl` | **absent** |
| `model/best_model_*/saved_model.pb` | present |
| `model/best_model_*/variables/` | **absent** |

See [reproduction.md](reproduction.md).

## Naming collisions worth remembering

* Notebook cells sometimes say `train_sentence.csv` (cwd-relative)
  and sometimes `dataset/train_sentence.csv`.
* SVM pickles are `svm_classifier.pkl` in one notebook and
  `svm_model.pkl` in the other.
* Deep models are `best_model_{single,multi}_modal` on disk and
  `best_model_{w,we}_<acc>_sub_<acc>` in the metrics notebook.
* `W` / `WE` in prose = single-modal / multi-modal in folder names.

## Suggested patches if you keep working on this repo

Keep them on a personal branch; this map is the checklist:

* add a real `requirements.txt` pinned to the Python 3.9 stack
* fix the except-blocks that construct `SVC()` for tree models
* size the embedding matrix with `len(word_index) + 1`
* save the Keras tokenizer next to the model
* put `variables/` in Git LFS or stop advertising `load_model`
