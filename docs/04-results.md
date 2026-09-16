# 04 — Results

All figures below are transcribed from executed cells in
`get_metrics_of_models.ipynb` (2023-06-05 / 2023-06-06) and
`evaluate_loaded_dl_models.ipynb`. They are **not** recomputed in this
documentation pass: the GloVe binary and the SavedModel weight shards are
not in git.

Notation:

* **W** — word / GloVe channel only
* **WE** — word + emoji2vec
* **Test** — 2,000 tweets, balanced
* **Subtest** — 278 emoji-containing tweets

Percentages are `100 ×` the notebook’s raw floats, rounded to two decimals
the same way the notebook’s `multiply_and_round` helper did.

## Accuracy

| Model | Test W | Test WE | Subtest W | Subtest WE |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 72.65 | 72.95 | 77.70 | 79.86 |
| SVM | 76.90 | 76.30 | 81.29 | 82.37 |
| Gradient boosting | 74.60 | 74.75 | 79.50 | 79.50 |
| Random forest | 81.45 | 81.80 | 80.58 | 85.25 |
| BiLSTM + attention | 86.35 | 87.35 | 86.69 | 89.21 |

## F1 (positive = sarcastic)

| Model | Test W | Test WE | Subtest W | Subtest WE |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 75.57 | 75.66 | 83.42 | 84.62 |
| SVM | 77.22 | 76.63 | 85.23 | 85.29 |
| Gradient boosting | 75.15 | 75.28 | 83.76 | 83.57 |
| Random forest | 82.32 | 82.55 | 85.25 | 88.39 |
| BiLSTM + attention | 86.56 | 86.86 | 89.40 | 91.07 |

GBT F1 is taken from the earlier notebook cells (`0.7515` / `0.7528` /
`0.8376` / `0.8357`); the compact `dl_list` block in the same notebook did
not include GBT.

## Precision and recall (sarcastic class)

From the later aggregation cell. GBT was not included in that loop.

| Model | Metric | Test W | Test WE | Subtest W | Subtest WE |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | recall | 78.30 | 77.70 | 87.21 | 82.56 |
| SVM | precision | 76.17 | 75.58 | 83.33 | 88.20 |
| DT | recall | 84.60 | 84.10 | 90.70 | 89.53 |
| DT | precision | 68.28 | 68.77 | 77.23 | 80.21 |
| RF | recall | 86.40 | 86.10 | 90.70 | 90.70 |
| RF | precision | 78.62 | 79.28 | 80.41 | 86.19 |
| BiLSTM | recall | 87.90 | 83.60 | 90.70 | 88.95 |
| BiLSTM | precision | 85.26 | 90.38 | 88.14 | 93.29 |

## Neural losses (evaluate notebook)

| Model | Test loss | Test acc | Subtest loss | Subtest acc |
| --- | ---: | ---: | ---: | ---: |
| best_model_single_modal (W) | 0.3118 | 0.8635 | 0.3056 | 0.8669 |
| best_model_multi_modal (WE) | 0.3200 | 0.8735 | 0.2852 | 0.8921 |

WE is slightly *worse* in test loss (0.320 vs 0.312) but better in
accuracy. On subtest both loss and accuracy move in the same direction
(emoji channel helps).

## What to take from this

1. **The neural model dominates the sklearn stack on every slice.** Order
   and attention matter more than the emoji channel.
2. **WE vs W on the full test set is a 1.0 point accuracy bump for the
   net, and ≤ 0.35 points for the trees / SVM.** That matches the 14%
   emoji rate: most rows cannot use the extra 200 dimensions.
3. **WE vs W on subtest is where the claim lives.** Random forest jumps
   4.67 points (80.58 → 85.25). The net jumps 2.52 points (86.69 → 89.21)
   and F1 89.40 → 91.07. SVM’s accuracy also ticks up, even though SVM
   *lost* 0.6 points on the full test set when emoji was concatenated —
   extra zeros on non-emoji tweets can be a nuisance for RBF SVM.
4. **Precision of the net on subtest WE is 93.29**, the highest cell in
   the table. Recall dips slightly vs W (88.95 vs 90.70). The emoji
   channel makes the net more conservative about the positive class on
   that slice, which is consistent with deadpan faces being a precision
   cue rather than a recall cue.
5. **Decision trees over-recall and under-precision** (test recall 84.6,
   precision 68.3). That is the usual unpruned-tree signature on a
   mildly imbalanced train set.

Machine-readable copy: [`metrics/published_metrics.csv`](metrics/published_metrics.csv).

## What this does *not* say

* It does not say emoji2vec beats a learned emoji embedding. The table
  was frozen.
* It does not say the gain would survive stripping `#sarcasm`. See
  `examples/03_lexical_cues.py` and `examples/06_bow_baseline.py`.
* It does not compare to a 2023 transformer. CCS2 was an RNN course
  project.
