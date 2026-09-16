# Results

Numbers below are copied from the executed cells in `get_metrics_of_models.ipynb` and `evaluate_loaded_dl_models.ipynb` (June 2023). They are **not** re-measured in this documentation pass: the GloVe dump and some sklearn pickles are not in the repo, and the Keras directories are missing `variables/`.

Column keys:

- **W / test** — word GloVe only, 2,000-tweet test set
- **WE / test** — word + emoji2vec, same test set
- **W / sub** — word only, 278-tweet subtest
- **WE / sub** — word + emoji2vec, subtest

## Accuracy (%)

| Model | W / test | WE / test | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 72.65 | 72.95 | 77.70 | 79.86 |
| SVM | 76.90 | 76.30 | 81.29 | 82.37 |
| Gradient boosting | 74.60 | 74.75 | 79.50 | 79.50 |
| Random forest | 81.45 | 81.80 | 80.58 | 85.25 |
| Bi-LSTM + attention | **86.35** | **87.35** | **86.69** | **89.21** |

## F1 (sarcastic class)

| Model | W / test | WE / test | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 75.57 | 75.66 | 83.42 | 84.62 |
| SVM | 77.22 | 76.63 | 85.23 | 85.29 |
| Gradient boosting | 75.15 | 75.28 | 83.76 | 83.57 |
| Random forest | 82.32 | 82.55 | 85.25 | 88.39 |
| Bi-LSTM + attention | **86.56** | **86.86** | **89.40** | **91.07** |

## Precision and recall (notebook dump)

`dl_list` in the metrics notebook stores `[accuracy, f1, recall, precision]` for the four columns above.

| Model | Split | Recall | Precision |
| --- | --- | ---: | ---: |
| SVM | W / test | 78.30 | 76.17 |
| SVM | WE / test | 77.70 | 75.58 |
| SVM | W / sub | 87.21 | 83.33 |
| SVM | WE / sub | 82.56 | 88.20 |
| Decision tree | W / test | 84.60 | 68.28 |
| Decision tree | WE / test | 84.10 | 68.77 |
| Decision tree | W / sub | 90.70 | 77.23 |
| Decision tree | WE / sub | 89.53 | 80.21 |
| Random forest | W / test | 86.40 | 78.62 |
| Random forest | WE / test | 86.10 | 79.28 |
| Random forest | W / sub | 90.70 | 80.41 |
| Random forest | WE / sub | 90.70 | 86.19 |
| Bi-LSTM + attention | W / test | 87.90 | 85.26 |
| Bi-LSTM + attention | WE / test | 83.60 | 90.38 |
| Bi-LSTM + attention | W / sub | 90.70 | 88.14 |
| Bi-LSTM + attention | WE / sub | 88.95 | 93.29 |

Gradient boosting only printed accuracy and F1 in that notebook (no precision/recall lists).

## Reading the table

1. **Sequence model beats mean-pooling.** Averaging GloVe throws away word order. The Bi-LSTM + attention stack is about five points above random forest on the full test set.
2. **Emoji fusion is a small lift on the i.i.d. test set** (+1.0 accuracy for the net, +0.35 for random forest) and a **larger lift on the emoji-rich subtest** (+2.5 accuracy for the net, +4.7 for random forest).
3. **SVM slightly *drops* when emoji is concatenated** on the full test set (76.90 → 76.30). Mean-pooled emoji zeros add 200 noisy dimensions for tweets with no in-vocab emoji.
4. **Decision trees over-recall and under-precision** on the full test set. That matches an unpruned `DecisionTreeClassifier()` on 200-D averages.
5. **Multimodal Bi-LSTM trades recall for precision** on the full test set (recall 87.90 → 83.60, precision 85.26 → 90.38). The F1 movement is small; the accuracy movement is the cleaner headline.

## Keras `evaluate` logs

From `evaluate_loaded_dl_models.ipynb`:

```text
multimodal  test    loss=0.3200  acc=0.8735
multimodal  subtest loss=0.2852  acc=0.8921
single      test    loss=0.3118  acc=0.8635
single      subtest loss=0.3056  acc=0.8669
```

Those accuracies match the table above.

## Lexical example baseline (this branch)

`examples/lexical_baseline.py` trains **without** GloVe: bag-of-words Naive Bayes, SGD logistic regression, and a hashtag/emoji cue rule. Full snapshot: [example_runs.md](example_runs.md). Headline numbers:

| Model | test acc | subtest acc |
| --- | ---: | ---: |
| Rule (sarcasm hashtag / stem+emoji) | 81.30% | 88.85% |
| Multinomial NB (unigrams) | 84.75% | 85.61% |
| SGD logistic (unigrams) | 87.95% | 89.93% |

SGD logistic matching the 2023 Bi-LSTM is explained in [findings.md](findings.md): the evaluation sets are dense in `#not`-style tags, so a bag-of-words model that keeps the raw hashtag is not a weak baseline. Mean-pooled GloVe SVM (76.9%) is weaker than counts for that reason.
