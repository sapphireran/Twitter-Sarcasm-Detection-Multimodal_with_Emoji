# Results

Numbers below are copied from the executed cells in
[`baseline_models.ipynb`](../baseline_models.ipynb),
[`evaluate_loaded_dl_models.ipynb`](../evaluate_loaded_dl_models.ipynb),
and [`get_metrics_of_models.ipynb`](../get_metrics_of_models.ipynb).
They are the 2023 course-project scores, not a fresh retrain.

Column order inside each cell of the notebooks is:

1. test, single-modal (word)
2. test, multi-modal (word + emoji)
3. subtest, single-modal
4. subtest, multi-modal

`test` is 2,000 balanced tweets. `subtest` is the 278-row emoji slice
of that same test set (see [dataset.md](dataset.md)).

## Test set (2,000 tweets)

| Model | Acc (word) | Acc (word+emoji) | F1 (word) | F1 (word+emoji) |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.769 | 0.763 | 0.772 | 0.766 |
| Decision Tree | 0.7265 | 0.7295 | 0.756 | 0.757 |
| Random Forest | 0.8145 | 0.818 | 0.823 | 0.826 |
| Gradient Boosting | 0.746 | 0.7475 | 0.751 | 0.753 |
| BiLSTM + Attention | **0.8635** | **0.8735** | **0.866** | **0.869** |

## Subtest (278 emoji-bearing tweets)

| Model | Acc (word) | Acc (word+emoji) | F1 (word) | F1 (word+emoji) |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.813 | 0.824 | 0.852 | 0.853 |
| Decision Tree | 0.777 | 0.799 | 0.834 | 0.846 |
| Random Forest | 0.806 | 0.853 | 0.852 | 0.884 |
| Gradient Boosting | 0.795 | 0.795 | 0.838 | 0.836 |
| BiLSTM + Attention | **0.867** | **0.892** | **0.894** | **0.911** |

## Precision and recall (same four columns)

From the printed lists at the bottom of `get_metrics_of_models.ipynb`.

| Model | Metric | test word | test +emoji | sub word | sub +emoji |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | recall | 0.783 | 0.777 | 0.872 | 0.826 |
| SVM | precision | 0.762 | 0.756 | 0.833 | 0.882 |
| Decision Tree | recall | 0.846 | 0.841 | 0.907 | 0.895 |
| Decision Tree | precision | 0.683 | 0.688 | 0.772 | 0.802 |
| Random Forest | recall | 0.864 | 0.861 | 0.907 | 0.907 |
| Random Forest | precision | 0.786 | 0.793 | 0.804 | 0.862 |
| BiLSTM + Attention | recall | 0.879 | 0.836 | 0.907 | 0.890 |
| BiLSTM + Attention | precision | 0.853 | 0.904 | 0.881 | 0.933 |

Gradient Boosting has accuracy and F1 in the notebook; it was not
included in that last printed precision/recall block.

## What the numbers say

* The BiLSTM + attention is ahead of every sklearn baseline on both
  splits, with or without emoji.
* On the **full test set**, adding emoji barely moves the baselines
  (sometimes down, as with SVM) and adds about one point of accuracy
  to the deep model (0.8635 → 0.8735). That is expected: only 13.9%
  of test tweets carry emoji, so a 400-d concat is a zero-vector in
  the emoji half most of the time.
* On the **emoji subtest**, the lift is larger. Random Forest jumps
  0.806 → 0.853 accuracy; the deep model 0.867 → 0.892 accuracy and
  0.894 → 0.911 F1. The multi-modal deep model also becomes more
  conservative (recall down, precision up) — it is less willing to
  call a tweet sarcastic unless the emoji channel agrees.

`examples/sarcasm_cues.py` is the place to re-count how many rows can
actually use the emoji half of the feature vector.

## Deep-model eval traces

`evaluate_loaded_dl_models.ipynb` recorded:

```text
multi-modal  test    loss 0.3200  acc 0.8735
multi-modal  subtest loss 0.2852  acc 0.8921
single-modal test    loss 0.3118  acc 0.8635
single-modal subtest loss 0.3056  acc 0.8669
```

Those SavedModels were later renamed to
`model/best_model_multi_modal` and `model/best_model_single_modal`.
The copies in the repository are missing their `variables/`
directories, so they will not load in a current TensorFlow without
the original training dump.
