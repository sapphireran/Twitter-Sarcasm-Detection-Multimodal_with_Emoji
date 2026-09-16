# Baselines and reported results

All figures in this file are **historical**. They were copied from the
executed outputs of `baseline_models.ipynb` and
`get_metrics_of_models.ipynb` (June 2023). This documentation pass did not
reload the Keras models or the sklearn pickles.

Machine-readable copy: [`reported_metrics.csv`](reported_metrics.csv).

## Protocol

- **Single-modal** classical: 200-d mean of GloVe Twitter hits.
- **Multi-modal** classical: 400-d concatenation of that mean and the mean
  of emoji2vec hits.
- **Single-modal** neural: frozen GloVe matrix, unknown tokens zero (emoji
  fill disabled).
- **Multi-modal** neural: unknown emoji tokens filled with the mean of
  their emoji2vec rows.
- **Test**: 2,000 tweets, 1,000 sarcastic.
- **Subtest**: 278 tweets, 172 sarcastic, almost all with emoji.

Classical models used the default sklearn constructors visible in
`baseline_models.ipynb` (`SVC()`, `DecisionTreeClassifier()`,
`RandomForestClassifier()`, `GradientBoostingClassifier()`). A few
fallback branches in that notebook construct the wrong estimator if a
pickle is missing (for example the random-forest single-modal branch
falls back to `SVC()`). The printed numbers below come from the
successful **load** path, not from those fallbacks.

## Test set (2,000 tweets)

| Model | Modal | Accuracy | F1 | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | single | 0.769 | 0.772 | 0.783 | 0.762 |
| SVM | multi | 0.763 | 0.766 | 0.777 | 0.756 |
| Decision tree | single | 0.727 | 0.756 | 0.846 | 0.683 |
| Decision tree | multi | 0.730 | 0.757 | 0.841 | 0.688 |
| Random forest | single | 0.815 | 0.823 | 0.864 | 0.786 |
| Random forest | multi | 0.818 | 0.826 | 0.861 | 0.793 |
| Gradient boosting | single | 0.746 | 0.751 | — | — |
| Gradient boosting | multi | 0.748 | 0.753 | — | — |
| BiLSTM + attention | single | 0.864 | 0.866 | 0.879 | 0.853 |
| BiLSTM + attention | multi | **0.874** | **0.869** | 0.836 | 0.904 |

Gradient-boosting precision/recall were not printed in the metrics
notebook; only accuracy and F1 were.

## Subtest (278 emoji-heavy tweets)

| Model | Modal | Accuracy | F1 | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | single | 0.813 | 0.852 | 0.872 | 0.833 |
| SVM | multi | 0.824 | 0.853 | 0.826 | 0.882 |
| Decision tree | single | 0.777 | 0.834 | 0.907 | 0.772 |
| Decision tree | multi | 0.799 | 0.846 | 0.895 | 0.802 |
| Random forest | single | 0.806 | 0.852 | 0.907 | 0.804 |
| Random forest | multi | 0.853 | 0.884 | 0.907 | 0.862 |
| Gradient boosting | single | 0.795 | 0.838 | — | — |
| Gradient boosting | multi | 0.795 | 0.836 | — | — |
| BiLSTM + attention | single | 0.867 | 0.894 | 0.907 | 0.881 |
| BiLSTM + attention | multi | **0.892** | **0.911** | 0.890 | 0.933 |

Neural precision/recall on the subtest come from `dl_list` in the metrics
notebook:

```
dl_list = [
  [0.8635, 0.8735, 0.8669, 0.8921],   # accuracy  test_s, test_m, sub_s, sub_m
  [0.8656, 0.8686, 0.8940, 0.9107],   # F1
  [0.8790, 0.8360, 0.9070, 0.8895],   # recall
  [0.8526, 0.9038, 0.8814, 0.9329],   # precision
]
```

## How to read the table

1. **Order beats averaging.** Every classical model is a bag of embeddings.
   The weakest neural result (86.4% test accuracy) still clears the
   strongest classical result (81.8%).
2. **Emoji help most where emoji exist.** Multi-modal minus single-modal
   accuracy on the main test set is about +0.3 (RF) to +1.0 (BiLSTM), and
   is slightly negative for SVM. On the subtest the same difference is
   +4.7 (RF) and +2.5 (BiLSTM). SVM also finally gains (+1.1).
3. **Precision/recall trade.** The multi-modal BiLSTM is more conservative
   on the main test set (recall 0.879 → 0.836, precision 0.853 → 0.904).
   On the subtest both F1 and precision rise.
4. **Hashtags still explain a lot.** A keyword rule that fires on `#not`,
   `#sarcasm`, `#sarcastictweet`, or `#yeahright` already covers a large
   sarcastic slice of the test set. See [`dataset.md`](dataset.md). The
   residual error is the interesting part: sarcastic tweets with no tag,
   and sincere tweets with a laughing emoji.

## Laptop baseline (this branch)

`examples/05_tfidf_baseline.py` trains a hashed-unigram logistic model on
the shipped CSVs. It is **not** a reimplementation of the GloVe SVM and
should not be compared to the table as if it were. It exists so a clone of
this repo can produce *a* accuracy number without 1.2 GB of GloVe.

Expect test accuracy in the mid-to-high 70s when the script uses the
default hash size and a few epochs. Hashtag features alone will carry a
lot of that number. The script prints a short ablation that zeros out
tokens starting with `#` so you can see the drop.

## Plots

The metrics notebook includes a bar chart of accuracies and F1 scores
(`Comparison of Accuracies and F1 Scores for Different Models`). The PNG
is embedded in the `.ipynb` and is not exported as a standalone file.
