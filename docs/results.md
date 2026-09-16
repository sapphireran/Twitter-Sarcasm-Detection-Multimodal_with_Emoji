# Results

Numbers below are copied from the saved outputs in
`get_metrics_of_models.ipynb` and `evaluate_loaded_dl_models.ipynb`
(June 2023). They are **not** re-measured in this docs pass: the
GloVe dump and SavedModel weight shards are not in git.

Notation:

- **W** — text / GloVe only.
- **WE** — text + emoji2vec.
- **full test** — 2,000 tweets, 1,000 / 1,000.
- **sub test** — 278 emoji-heavy tweets, 106 / 172.

## Accuracy

| Model | W full | WE full | W sub | WE sub |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| SVM | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| Gradient boosting | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| Random forest | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| Bi-LSTM + attention | 0.8635 | **0.8735** | 0.8669 | **0.8921** |

## F1 (positive = sarcastic)

| Model | W full | WE full | W sub | WE sub |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| SVM | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Gradient boosting | 0.7515 | 0.7528 | 0.8376 | 0.8357 |
| Random forest | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| Bi-LSTM + attention | 0.8656 | **0.8686** | 0.8940 | **0.9107** |

## Precision and recall (from the notebook’s hardcoded lists)

Order of each quartet: W full, WE full, W sub, WE sub.

**SVM**

| Metric | W full | WE full | W sub | WE sub |
| --- | ---: | ---: | ---: | ---: |
| Recall | 0.783 | 0.777 | 0.872 | 0.826 |
| Precision | 0.762 | 0.756 | 0.833 | 0.882 |

**Decision tree**

| Metric | W full | WE full | W sub | WE sub |
| --- | ---: | ---: | ---: | ---: |
| Recall | 0.846 | 0.841 | 0.907 | 0.895 |
| Precision | 0.683 | 0.688 | 0.772 | 0.802 |

**Random forest**

| Metric | W full | WE full | W sub | WE sub |
| --- | ---: | ---: | ---: | ---: |
| Recall | 0.864 | 0.861 | 0.907 | 0.907 |
| Precision | 0.786 | 0.793 | 0.804 | 0.862 |

**Bi-LSTM + attention** (`dl_list` cell)

| Metric | W full | WE full | W sub | WE sub |
| --- | ---: | ---: | ---: | ---: |
| Recall | 0.879 | 0.836 | 0.907 | 0.890 |
| Precision | 0.853 | 0.904 | 0.881 | 0.933 |

The deep multi-modal model is the only one that **drops recall** on
the full test set while raising precision. That is consistent with
emoji rows acting as a conservative extra vote: some sarcastic tweets
without useful emoji are no longer pushed over 0.5, while the ones
with a matching emoji / tag become more confidently positive.

## Deep model losses (evaluate notebook)

| Checkpoint | Test loss / acc | Subtest loss / acc |
| --- | --- | --- |
| `best_model_single_modal` | 0.3118 / 0.8635 | 0.3056 / 0.8669 |
| `best_model_multi_modal` | 0.3200 / 0.8735 | 0.2852 / 0.8921 |

The multi-modal checkpoint has a slightly **higher** test loss and a
**lower** subtest loss. Accuracy still goes up on both splits.

## Reading the tables

1. Every model beats chance (0.50 on full test). Even the decision
   tree is in the low 0.70s.
2. Emoji concat is close to a wash on the full test set for SVM
   (accuracy falls 0.006) and only a small gain for trees / GBT.
3. The gain shows up on the subtest, especially random forest
   (+0.047 acc) and Bi-LSTM (+0.025 acc).
4. Sequence models beat mean-pooled bags by a wide margin. Word
   order and a late `#not` are exactly the pattern attention can
   lock onto.
5. Do not treat the subtest as a second i.i.d. test. It is an
   emoji-conditioned slice with a different label prior.

## A lexical floor (this docs pass)

`examples/lexical_sarcasm_baseline.py` fits a NumPy logistic
regression on surface features only (hashtag flags, emoji counts,
elongation, punctuation). It is a **lower bound** for how much of
the test set is solvable without GloVe. Run it locally; the number
moves with the random seed because the train shuffle is seeded in
the script. On a typical run the hashtag flags alone already clear
the mid-0.60s on full test — that is the `#not` / `#sarcasm`
artifact described in [dataset.md](dataset.md).
