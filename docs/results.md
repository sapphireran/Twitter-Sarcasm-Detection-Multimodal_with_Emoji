# Recorded 2023 results

Copied from `get_metrics_of_models.ipynb` and
`evaluate_loaded_dl_models.ipynb` (June 2023). These are not re-runs.

W = word / GloVe only. WE = word + emoji2vec.

## Official test (n = 2,000, balanced)

| Model | Acc W | Acc WE | F1 W | F1 WE | Prec W | Rec W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.7690 | 0.7630 | 0.7722 | 0.7663 | 0.7617 | 0.7830 |
| Decision tree | 0.7265 | 0.7295 | 0.7557 | 0.7566 | 0.6828 | 0.8460 |
| Random forest | 0.8145 | 0.8180 | 0.8232 | 0.8255 | 0.7862 | 0.8640 |
| Gradient boosting | 0.7460 | 0.7475 | 0.7515 | 0.7528 | — | — |
| BiLSTM + attention | 0.8635 | **0.8735** | 0.8656 | 0.8686 | — | — |

## Subtest (n = 278, emoji slice of test)

| Model | Acc W | Acc WE | F1 W | F1 WE |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.8129 | 0.8237 | 0.8523 | 0.8529 |
| Decision tree | 0.7770 | 0.7986 | 0.8342 | 0.8462 |
| Random forest | 0.8058 | 0.8525 | 0.8525 | 0.8839 |
| Gradient boosting | 0.7950 | 0.7950 | 0.8376 | 0.8357 |
| BiLSTM + attention | 0.8669 | **0.8921** | 0.8940 | 0.9107 |

Subtest precision / recall for the sklearn models (W, then WE):

| Model | P W | P WE | R W | R WE |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.8333 | 0.8820 | 0.8721 | 0.8256 |
| Decision tree | 0.7723 | 0.8021 | 0.9070 | 0.8953 |
| Random forest | 0.8041 | 0.8619 | 0.9070 | 0.9070 |

## How to read the table

* The LSTM is the only model that clearly beats a strong bag-of-words
  forest on test.
* WE never hurts the LSTM. It sometimes hurts SVM on test (zeros on
  the emoji half of the 400d vector).
* Every model except RF-W is higher on subtest than on test. Subtest
  is cue-dense *and* emoji-dense, and it is 61.9% sarcastic rather
  than 50%.
* Decision trees have high recall and low precision: they over-predict
  sarcastic, which is cheap on a balanced test set and looks even
  better on the sarcastic-heavy subtest.

Reprint the same numbers with `python3 examples/07_reprint_scores.py`.
