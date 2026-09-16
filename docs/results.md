# Recorded results (June 2023)

All figures below are copied from the executed notebooks
`get_metrics_of_models.ipynb` and `evaluate_loaded_dl_models.ipynb`
(5–6 June 2023). They are **not** recomputed in CI. The same numbers
live in `examples/lib/metrics.py` and can be reprinted with:

```bash
python3 examples/report_results.py --all
```

Column key:

* **W test** — word / GloVe only, official 2,000-tweet test
* **WE test** — GloVe + emoji2vec, official test
* **W subtest** — word only, 278-tweet emoji-rich slice
* **WE subtest** — GloVe + emoji2vec, same slice

Percentages are notebook values × 100, rounded to two decimals.

## Accuracy

| Model | W test | WE test | W subtest | WE subtest |
| --- | ---: | ---: | ---: | ---: |
| SVM | 76.90 | 76.30 | 81.29 | 82.37 |
| Decision Tree | 72.65 | 72.95 | 77.70 | 79.86 |
| Random Forest | 81.45 | 81.80 | 80.58 | 85.25 |
| Gradient Boosting | 74.60 | 74.75 | 79.50 | 79.50 |
| BiLSTM + Attention | 86.35 | **87.35** | 86.69 | **89.21** |

## F1 (positive = sarcastic)

| Model | W test | WE test | W subtest | WE subtest |
| --- | ---: | ---: | ---: | ---: |
| SVM | 77.22 | 76.63 | 85.23 | 85.29 |
| Decision Tree | 75.57 | 75.66 | 83.42 | 84.62 |
| Random Forest | 82.32 | 82.55 | 85.25 | 88.39 |
| Gradient Boosting | 75.15 | 75.28 | 83.76 | 83.57 |
| BiLSTM + Attention | 86.56 | 86.86 | 89.40 | **91.07** |

## Precision / recall (where the notebook exported them)

Test set, WE column unless noted.

| Model | P (W test) | P (WE test) | R (W test) | R (WE test) |
| --- | ---: | ---: | ---: | ---: |
| SVM | 76.17 | 75.58 | 78.30 | 77.70 |
| Decision Tree | 68.28 | 68.77 | 84.60 | 84.10 |
| Random Forest | 78.62 | 79.28 | 86.40 | 86.10 |
| BiLSTM + Attention | 85.26 | **90.38** | 87.90 | 83.60 |

The deep WE model is the precision specialist on the official test
set: emoji geometry cuts false positives more than it helps recall
(recall actually drops from 87.9% to 83.6%). On the subtest the same
model is both precise (93.29%) and still high-recall (88.95%).

Gradient boosting has no exported precision / recall in the notebook
summary cell; only accuracy and F1 were stored.

## Rule baseline (computed from the CSVs in this snapshot)

`python3 examples/sarcasm_cues.py --split test` predicts sarcastic if
the tweet has an explicit marker (`#not`, `#sarcasm`, `#yeahright`, …)
or a positive word plus a groan emoji. On the official test set:

| | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| surface-cue rule | 80.80 | 99.80 | 61.70 | 76.30 |

613 / 1,000 sarcastic test tweets carry an explicit marker. `#not`
(465), `#sarcasm` (89), `#sarcastictweet` (38), and `#yeahright` (19)
are all labelled sarcastic in this split. The rule therefore matches
random forest on accuracy but with almost no false positives and a
large recall hole — the BiLSTM's extra 6–7 points have to come from
tweets that look sincere until the whole sequence is read.

## How to read the W vs WE gap

1. **Official test is almost saturated by words.** WE adds +1.0 point
   of accuracy for the BiLSTM, +0.35 for forest, and *hurts* SVM
   slightly (−0.60). If you only look at this column you could
   conclude emoji2vec is optional.
2. **The subtest is where emoji identity pays rent.** Forest jumps
   +4.67 accuracy; BiLSTM +2.52; SVM +1.08. That slice was chosen
   because it is emoji-heavy, so this is a capability check, not a
   new official leaderboard.
3. **Trees over-predict sarcastic.** Decision-tree recall sits at
   84–90% with precision in the high 60s / low 70s on test. That
   matches a stump that has discovered `#not` and a few groan faces
   and then over-fires.
4. **The BiLSTM is the only model in the 86–87% band.** The
   architecture (two BiLSTMs + attention, frozen 200-d embeddings) is
   doing work that mean-pooling cannot: it can put mass on `#not`
   *after* reading "I love walking to school".
5. **A hashtag rule is not the whole story.** The surface-cue baseline
   already hits 80.8% test accuracy at 99.8% precision. That is a
   strong prior in *this* dump, and it is also why the remaining gap
   up to 87.35% is the interesting part of the project.

## Saved-model filenames vs folders

The metrics notebook loaded:

```
model/best_model_w_0.8634999990463257_sub_0.866906464099884
model/best_model_we_0.8734999895095825_sub_0.8920863270759583
```

The snapshot on GitHub instead has:

```
model/best_model_single_modal/
model/best_model_multi_modal/
```

`evaluate_loaded_dl_models.ipynb` evaluates those two folders and
prints the same 0.8635 / 0.8735 / 0.8669 / 0.8921 accuracies. Treat
them as the same two runs under shorter names. **Weight shards are
not in the snapshot** — only `saved_model.pb` and
`keras_metadata.pb`. You cannot `tf.keras.models.load_model` these
folders as-is.

## What a new experiment should report

If you add a model, fill the same four cells (W/WE × test/subtest)
and the same four metrics (accuracy, F1, precision, recall). Add the
row to `examples/lib/metrics.py` so `report_results.py` and this page
stay in sync. Do not drop the W column: the whole point of the
project is the delta.
