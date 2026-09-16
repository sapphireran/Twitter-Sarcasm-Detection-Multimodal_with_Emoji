# Experiments and recorded results

All numbers below are copied from the executed cells in
`baseline_models.ipynb` and `get_metrics_of_models.ipynb` (5–6 June 2023).
They are also stored as data in `examples/results_catalog.py` so scripts
and docs cannot drift independently.

Column order is always:

1. test set, single-modal (word / GloVe only)
2. test set, multi-modal (word + emoji)
3. subtest, single-modal
4. subtest, multi-modal

The subtest is the 278 emoji-containing test tweets. See
[data-pipeline.md](data-pipeline.md).

## Accuracy

| Model | Test word | Test word+emoji | Subtest word | Subtest word+emoji |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 72.65 | 72.95 | 77.70 | 79.86 |
| SVM | 76.90 | 76.30 | 81.29 | 82.37 |
| Gradient boosting | 74.60 | 74.75 | 79.50 | 79.50 |
| Random forest | 81.45 | 81.80 | 80.58 | 85.25 |
| Bi-LSTM + attention | **86.35** | **87.35** | **86.69** | **89.21** |

## F1 (positive = sarcastic)

| Model | Test word | Test word+emoji | Subtest word | Subtest word+emoji |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 75.57 | 75.66 | 83.42 | 84.62 |
| SVM | 77.22 | 76.63 | 85.23 | 85.29 |
| Gradient boosting | 75.15 | 75.28 | 83.76 | 83.57 |
| Random forest | 82.32 | 82.55 | 85.25 | 88.39 |
| Bi-LSTM + attention | **86.56** | **86.86** | **89.40** | **91.07** |

## Precision and recall (Bi-LSTM)

| Setting | Precision | Recall |
| --- | ---: | ---: |
| Test, word | 85.26 | 87.90 |
| Test, word+emoji | 90.38 | 83.60 |
| Subtest, word | 88.14 | 90.70 |
| Subtest, word+emoji | 93.29 | 88.95 |

On the full test set the emoji-aware LSTM **raises precision and lowers
recall**. It becomes more conservative about calling a tweet sarcastic,
which is consistent with emoji sometimes being sincere (😭 after bad news)
and sometimes being the sarcasm marker. The net accuracy still goes up
because the test set is balanced.

The classical models do not show that precision/recall trade-off as
cleanly. Random Forest keeps recall at 86.1–86.4% on test and 90.7% on
subtest in both modes; the emoji channel mainly lifts precision on the
emoji subset (80.41% → 86.19%).

## Full classical metric cards

Values are fractions, not percents.

### SVM

| Metric | Test w | Test we | Sub w | Sub we |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| F1 | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Recall | 0.7830 | 0.7770 | 0.8721 | 0.8256 |
| Precision | 0.7617 | 0.7558 | 0.8333 | 0.8820 |

### Decision tree

| Metric | Test w | Test we | Sub w | Sub we |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| F1 | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| Recall | 0.8460 | 0.8410 | 0.9070 | 0.8953 |
| Precision | 0.6828 | 0.6877 | 0.7723 | 0.8021 |

### Random forest

| Metric | Test w | Test we | Sub w | Sub we |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| F1 | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| Recall | 0.8640 | 0.8610 | 0.9070 | 0.9070 |
| Precision | 0.7862 | 0.7928 | 0.8041 | 0.8619 |

### Gradient boosting

Only accuracy and F1 were printed as named scores in the notebook. The
accuracy pair on subtest is identical to three decimals in both modes
(0.79496), so boosting did not use the extra 200 dimensions on that slice.

| Metric | Test w | Test we | Sub w | Sub we |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| F1 | 0.7515 | 0.7528 | 0.8376 | 0.8357 |

## How to read the headline

Three facts are robust across the recorded run:

1. **Recurrence plus attention beats mean pooling.** The weakest LSTM
   setting (86.35% test accuracy) is still 4.9 points above the strongest
   classical setting (Random Forest at 81.45%).
2. **Emoji features are not free lunch on the full test set.** Most test
   tweets have no emoji, so the extra channel is zeros. SVM gets slightly
   worse. Trees and the LSTM improve only a little.
3. **The emoji channel matters when emoji are present.** Restricting to
   the 278-tweet subtest turns a +1.0 LSTM accuracy gap into a +2.5 gap,
   and turns a +0.35 Random Forest gap into a +4.7 gap.

That is the argument the course project was built to make: evaluate
multi-modal social-media models on the subset that actually carries the
second modality, not only on a mixed official split.

## What was not measured

- No confidence intervals or McNemar tests on the paired predictions
- No ablation of `#not` / `#sarcasm` (those cues alone are strong)
- No per-emoji error analysis
- No training curves in the repo
- No hyperparameter search

The lexical baseline in `examples/lexical_baseline.py` is a *new* personal
walkthrough. It uses hand-built surface features and a NumPy logistic
model so you can get a number without GloVe. It is not a 2023 result and
must not be mixed into the table above.
