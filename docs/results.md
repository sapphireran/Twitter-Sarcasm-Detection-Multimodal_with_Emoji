# Results

Numbers below are copied from executed cells in
`get_metrics_of_models.ipynb` and `baseline_models.ipynb` (5–6 June 2023).
The typed copy used by the examples is
[`examples/reported_results.json`](../examples/reported_results.json).

`single` = GloVe only. `multi` = GloVe + emoji2vec. Positive class =
sarcastic.

## Test set (n = 2,000, balanced)

| Model | Mode | Acc | F1 | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | single | 0.7690 | 0.7722 | 0.783 | 0.7617 |
| SVM | multi | 0.7630 | 0.7663 | 0.777 | 0.7558 |
| Decision tree | single | 0.7265 | 0.7557 | 0.846 | 0.6828 |
| Decision tree | multi | 0.7295 | 0.7566 | 0.841 | 0.6877 |
| Random forest | single | 0.8145 | 0.8232 | 0.864 | 0.7862 |
| Random forest | multi | 0.8180 | 0.8255 | 0.861 | 0.7928 |
| Gradient boosting | single | 0.7460 | 0.7515 | — | — |
| Gradient boosting | multi | 0.7475 | 0.7528 | — | — |
| BiLSTM + attention | single | 0.8635 | 0.8656 | 0.879 | 0.8526 |
| **BiLSTM + attention** | **multi** | **0.8735** | **0.8686** | 0.836 | **0.9038** |

On the balanced test set, emoji2vec is a small win for trees/forests and
about +1.0 accuracy for the BiLSTM. SVM actually drops slightly when
emoji is concatenated — mean-pooling emoji into a 400-d vector is a
weaker use of that channel than the embedding-table mix in `Preprocess`.

## Subtest (n = 278, 99.3% contain emoji)

| Model | Mode | Acc | F1 | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | single | 0.8129 | 0.8523 | 0.8721 | 0.8333 |
| SVM | multi | 0.8237 | 0.8529 | 0.8256 | 0.8820 |
| Decision tree | single | 0.7770 | 0.8342 | 0.9070 | 0.7723 |
| Decision tree | multi | 0.7986 | 0.8462 | 0.8953 | 0.8021 |
| Random forest | single | 0.8058 | 0.8525 | 0.9070 | 0.8041 |
| Random forest | multi | 0.8525 | 0.8839 | 0.9070 | 0.8619 |
| Gradient boosting | single | 0.7950 | 0.8376 | — | — |
| Gradient boosting | multi | 0.7950 | 0.8357 | — | — |
| BiLSTM + attention | single | 0.8669 | 0.8940 | 0.9070 | 0.8814 |
| **BiLSTM + attention** | **multi** | **0.8921** | **0.9107** | 0.8895 | **0.9329** |

This is where multi-modal fusion pays off: random forest +4.7 acc,
BiLSTM +2.5 acc, and the best F1 in the project (0.911).

Keras `evaluate` on the saved models (same notebook) reported:

* multi: test loss 0.3200 / acc 0.8735; subtest loss 0.2852 / acc 0.8921
* single: test loss 0.3118 / acc 0.8635; subtest loss 0.3056 / acc 0.8669

## Cue floor measured by the new examples

`examples/04_lexicon_baseline.py` (predict sarcastic iff a cue hashtag is
present) on this checkout:

| Split | Acc | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.6241 | 0.9753 | 0.1962 | 0.3267 |
| test | 0.8070 | 1.0000 | 0.6140 | 0.7608 |
| subtest | 0.8633 | 1.0000 | 0.7791 | 0.8758 |

Precision is essentially perfect: if the tweet says `#not` / `#sarcasm`,
it is sarcastic in this dump. Recall is the tell — only 19.6% of
*train* sarcastic tweets wear the tag, vs 61.4% of test and 77.9% of
subtest. The BiLSTM’s test accuracy (0.8735 multi) still clears this
0.807 lexicon floor; that gap is the part that is not just reading the
hashtag. The BiLSTM is more interesting on sarcastic tweets that have
no cue tag (see the train examples in [dataset.md](dataset.md)).

## Plot

`get_metrics_of_models.ipynb` has a grouped bar chart titled
“Comparison of Accuracies and F1 Scores for Different Models” with
bar width 0.15. Re-run that cell if you want the PNG; it is stored
inline in the notebook, not as a standalone file.
