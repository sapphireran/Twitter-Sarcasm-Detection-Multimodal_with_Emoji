# Results

Numbers in the first two tables are taken from the executed cells in
`get_metrics_of_models.ipynb` and `baseline_models.ipynb`. They describe the
saved checkpoints, not a fresh retrain. The heuristic table is computed by
`examples/heuristic_baseline.py` against the files in `dataset/`.

`W` is the single-modal (word-only) model. `WE` is the multi-modal model that
is allowed to fill OOV tokens from emoji2vec. "Subtest" is the 278-tweet
emoji slice of the official test set; read `docs/dataset.md` before citing it
as generalisation.

## Accuracy

| Model | Test W | Test WE | Subtest W | Subtest WE |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| SVM | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| Gradient Boosting | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| Random Forest | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| Bi-LSTM + attention | 0.8635 | **0.8735** | 0.8669 | **0.8921** |

## F1 (sarcastic class)

| Model | Test W | Test WE | Subtest W | Subtest WE |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| SVM | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Random Forest | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| Bi-LSTM + attention | 0.8656 | **0.8686** | 0.8940 | **0.9107** |

Gradient Boosting F1 was not printed as a standalone block in the notebook;
its accuracy is in the table above.

## Precision and recall (notebook dump)

The last metrics cell in `get_metrics_of_models.ipynb` also stored precision
and recall for the classical models (order: test W, test WE, subtest W,
subtest WE).

| Model | Metric | Test W | Test WE | Subtest W | Subtest WE |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | Recall | 0.783 | 0.777 | 0.872 | 0.826 |
| SVM | Precision | 0.762 | 0.756 | 0.833 | 0.882 |
| Decision Tree | Recall | 0.846 | 0.841 | 0.907 | 0.895 |
| Decision Tree | Precision | 0.683 | 0.688 | 0.772 | 0.802 |
| Random Forest | Recall | 0.864 | 0.861 | 0.907 | 0.907 |
| Random Forest | Precision | 0.786 | 0.793 | 0.804 | 0.862 |
| Bi-LSTM + ATT | Recall | 0.879 | 0.836 | 0.907 | 0.890 |
| Bi-LSTM + ATT | Precision | 0.853 | 0.904 | 0.881 | 0.933 |

The multi-modal Bi-LSTM trades a little recall for a large precision gain on
the full test set (0.853 → 0.904). That is the cleanest evidence in the
original run that emoji2vec is doing something other than "predict sarcastic
whenever a face appears".

## What emoji2vec actually moved

On the official test set the WE increment is small for every classical model
and about +1.0 accuracy point for the Bi-LSTM (0.8635 → 0.8735). The jump
looks bigger on subtest because that split *is* the emoji tweets. Random
Forest WE on subtest (0.8525) is the best classical score; it still trails
the multi-modal recurrent model by four accuracy points.

SVM WE is slightly *worse* than SVM W on the full test set (0.763 vs 0.769).
Averaging emoji2vec into a 400-d bag is not free: tweets with no emoji pick
up a zero block, and tweets with a single decorative glyph can move the mean
in a way the linear-ish SVM does not like.

## Explicit-cue heuristic

`sarcasm_lib.heuristic` predicts sarcastic when a tweet has a marker hashtag
or a positive opener stacked on a negative emoji. No embeddings.

| Split | Accuracy | Precision | Recall | F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 0.6254 | 0.9633 | 0.2018 | 0.3336 | 3730 / 142 / 14758 |
| test | 0.8115 | 1.0000 | 0.6230 | 0.7677 | 623 / 0 / 377 |
| subtest | 0.8885 | 1.0000 | 0.8198 | 0.9010 | 141 / 0 / 31 |

Precision is 1.0 on both evaluation splits: if the heuristic fires, the gold
label is sarcastic. Recall is the whole story. On the full test set the
Bi-LSTM still recovers the 377 sarcastic tweets that do not wear `#not` on
their sleeve. On subtest the heuristic F1 (0.901) is already next to the
multi-modal network (0.911), which is a warning about that split, not a
reason to retire the network.

## How to quote this project

* Quote **test WE accuracy 0.8735 / F1 0.8686** as the headline neural
  result.
* Quote the **+1.0 test accuracy** gap between Bi-LSTM W and WE as the
  emoji-ablation result.
* Do not quote subtest 0.89 as "emoji-only generalisation" without saying
  the tweets are a subset of test and are saturated with `#not`.
