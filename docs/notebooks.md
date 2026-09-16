# Original notebooks

Three Jupyter notebooks are the 2023 lab trail. They are kept as-is. This page is a table of contents so you do not have to scroll 100 KB of plot PNG to find a number.

## `baseline_models.ipynb`

Purpose: train or reload four sklearn families on mean-pooled features.

| Cells (approx.) | What they do |
| --- | --- |
| imports | `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, gensim, joblib, `ml_read_data` |
| load vectors | `glove.twitter.27B.200d.bin`, `emoji2vec_twitter.bin` |
| load splits | `ml_read_data` on `train_` / `test_` / `subtest_` CSVs **in the repo root** |
| SVM / DT / RF / GBT blocks | `joblib.load` if present, else `fit` and dump under `baseline_models/` |
| score cells | accuracy on test and subtest, word vs word+emoji |

Printed accuracies match the metrics notebook:

- SVM 0.769 / 0.763 / 0.813 / 0.824
- DT 0.7265 / 0.7295 / 0.777 / 0.799
- RF 0.8145 / 0.818 / 0.806 / 0.853
- GBT 0.746 / 0.7475 / 0.795 / 0.795

Caveat: several `except FileNotFoundError` branches construct `SVC()` under the wrong heading. If the pickle load succeeds, those branches never run. See [architecture.md](architecture.md).

## `evaluate_loaded_dl_models.ipynb`

Purpose: score the two saved Keras models.

| Cells | What they do |
| --- | --- |
| imports | TensorFlow, gensim, `Preprocess`, `preprocess_test`, `ReadOpen` |
| load vectors | same binaries as the baseline notebook |
| preprocess | train-fit tokenizer + embedding matrix; pad test / subtest to train `maxlen` |
| `best_model_multi_modal` | `evaluate` → test **0.8735**, subtest **0.8921** |
| `summary()` | 2,510,848 params, length 78, width 200 / 512 |
| `best_model_single_modal` | `evaluate` → test **0.8635**, subtest **0.8669** |
| `summary()` | same graph |

The notebook does not compute F1; that lives next door.

## `get_metrics_of_models.ipynb`

Purpose: one place for accuracy, F1, precision, recall, and a bar chart.

| Section | What it does |
| --- | --- |
| load vectors | `glove_tt.txt` (**text**, not binary) + `emoji2vec_twitter.bin` |
| sklearn features | `ml_read_data` with `dataset/…` paths |
| SVM / DT / RF / GBT | load pickles (`svm_model.pkl` naming differs from `svm_classifier.pkl` in the other notebook) |
| neural preprocess | `ReadOpen` + `Preprocess(..., get_emoji2vec=True)` |
| neural scores | print BiLSTM+Attn acc and F1 for both models |
| plot | grouped bars of acc / F1 |
| metric dump | builds `*_accuracy_score_list` etc. and prints them |

Sklearn pickle **filenames do not match** across notebooks (`svm_model.pkl` vs `svm_classifier.pkl`). Whichever names existed on the 2023 laptop are what produced the printed numbers. This clone has `dt_classifier.pkl` / `gbt_classifier.pkl` only.

The huge PNG in the notebook is that comparison plot; it is not a network diagram.

## Recommended reading order

1. This page (where the numbers live).
2. [experiments.md](experiments.md) (what the numbers mean).
3. [preprocessing.md](preprocessing.md) + [architecture.md](architecture.md).
4. `examples/` if you want to touch the ideas without restoring TensorFlow.
