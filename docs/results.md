# Recorded results

All figures in this file are **transcribed from executed notebook outputs** already in the repository (`baseline_models.ipynb`, `evaluate_loaded_dl_models.ipynb`, `get_metrics_of_models.ipynb`). They are not re-scored from the Keras checkpoints in this environment.

Column key:

- **W / full** — text-only representation, 2,000-row test set
- **WE / full** — text + emoji representation, 2,000-row test set
- **W / sub** — text-only, 278-row emoji-rich subtest
- **WE / sub** — text + emoji, 278-row subtest

## Accuracy

| Model | W / full | WE / full | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| SVM | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| Gradient boosting | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| Random forest | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| BiLSTM + attention | 0.8635 | 0.8735 | 0.8669 | 0.8921 |

Keras `evaluate()` on the saved checkpoints matches the last row: multimodal test loss 0.3200 / acc 0.8735; multimodal subtest loss 0.2852 / acc 0.8921; text-only test loss 0.3118 / acc 0.8635; text-only subtest loss 0.3056 / acc 0.8669.

## F1 (positive = sarcastic)

| Model | W / full | WE / full | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| SVM | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Random forest | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| BiLSTM + attention | 0.8656 | 0.8686 | 0.8940 | 0.9107 |

Gradient-boosting F1 was computed in the notebook for accuracy only in the early cells; the later dump that stores `dl_list` does not include a GBT block.

## Recall and precision (sklearn dump)

From the cell that printed `*_list` arrays in `get_metrics_of_models.ipynb`. Order inside each vector is `[W/full, WE/full, W/sub, WE/sub]`.

### SVM

| Metric | W / full | WE / full | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| F1 | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Recall | 0.7830 | 0.7770 | 0.8721 | 0.8256 |
| Precision | 0.7617 | 0.7558 | 0.8333 | 0.8820 |

### Decision tree

| Metric | W / full | WE / full | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| F1 | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| Recall | 0.8460 | 0.8410 | 0.9070 | 0.8953 |
| Precision | 0.6828 | 0.6877 | 0.7723 | 0.8021 |

### Random forest

| Metric | W / full | WE / full | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| F1 | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| Recall | 0.8640 | 0.8610 | 0.9070 | 0.9070 |
| Precision | 0.7862 | 0.7928 | 0.8041 | 0.8619 |

### BiLSTM + attention

| Metric | W / full | WE / full | W / sub | WE / sub |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.8635 | 0.8735 | 0.8669 | 0.8921 |
| F1 | 0.8656 | 0.8686 | 0.8940 | 0.9107 |
| Recall | 0.8790 | 0.8360 | 0.9070 | 0.8895 |
| Precision | 0.8526 | 0.9038 | 0.8814 | 0.9329 |

The multimodal LSTM trades some recall for a large precision jump on the full test set (0.853 → 0.904). On the emoji-rich subtest it gains both F1 and accuracy.

## Reading the deltas

`Δ = WE − W` on accuracy:

| Model | Δ full | Δ sub |
| --- | ---: | ---: |
| Decision tree | +0.003 | +0.022 |
| SVM | −0.006 | +0.011 |
| Gradient boosting | +0.001 | 0.000 |
| Random forest | +0.004 | +0.047 |
| BiLSTM + attention | +0.010 | +0.025 |

Concatenated emoji averages are close to noise on the full test set (emoji are absent from ~86% of rows). They become useful when every row has an emoji. Sequence-level fusion is the only setting where the full test set also moves by a full point.

## What a lexical baseline should beat — and what it should not

A hashtag-aware bag-of-tokens model will look strong because the labels themselves were largely hashtag-derived. `examples/lexical_baseline.py` is that model. Compare it to the table above as a **ceiling on “read the tag” performance**, not as a competitor to BiLSTM + GloVe.

A fair “does emoji2vec help?” claim has to hold after those tags are stripped. The original notebooks do not report a tag-ablated number. The example script prints both the full-feature score and a `--strip-supervision-tags` score so you can see the drop.

Regenerated in this checkout (stdlib + NumPy Naive Bayes, no GloVe):

| System | Test acc | Test F1 | Subtest acc | Subtest F1 |
| --- | ---: | ---: | ---: | ---: |
| Supervision-tag rule (`#not` / `#sarcasm` / …) | 0.807 | 0.760 | 0.863 | 0.876 |
| Naive Bayes (tags kept) | 0.833 | 0.852 | 0.871 | 0.904 |
| Naive Bayes (tags stripped) | 0.744 | 0.772 | 0.712 | 0.777 |

The tag rule has **perfect precision** on the test and subtest files (613 / 2,000 test tweets carry a supervision hashtag; all of them are labeled 1). Hiding those tags drops NB test accuracy by about 9 points. That gap is the part of the task that is “just reading the tag.” See [lexical_baseline.md](lexical_baseline.md).

## Plot

`get_metrics_of_models.ipynb` contains a grouped bar chart of accuracy plus overlaid F1 lines (`Comparison of Accuracies and F1 Scores for Different Models`). The PNG is stored inside the notebook, not as a standalone file.
