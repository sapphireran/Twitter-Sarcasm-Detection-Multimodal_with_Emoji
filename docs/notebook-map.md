# Notebook map

The three Jupyter notebooks are the original 2023 experiment log. They were
not cleaned for publication: cells are numbered out of order, some paths
disagree, and a few fallback constructors are copy-paste errors. This map
says what to open for which question.

## `baseline_models.ipynb`

**Purpose.** Train or reload the four sklearn baselines in both fusion
modes and print accuracy.

**Inputs.**

- `glove.twitter.27B.200d.bin`
- `emoji2vec_twitter.bin`
- `train_sentence.csv` / `train_label.csv` (cwd, not `dataset/`)
- matching test and subtest files in cwd

**Outputs.**

- `baseline_models/svm_classifier.pkl` and `svm_classifier_we.pkl`
- `dt_classifier.pkl` / `dt_classifier_we.pkl`
- `rf_classifier.pkl` / `rf_classifier_we.pkl`
- `gbt_classifier.pkl` / `gbt_classifier_we.pkl`

The repository currently contains only the decision-tree and gradient-boosting
pickles. SVM and random-forest files were never pushed (or were later
removed). The executed accuracy cells remain in the notebook.

**Watch-outs.**

- The decision-tree / random-forest / GBT `except FileNotFoundError`
  branches construct an `SVC()` for at least one of the two modes. Ignore
  those branches if you retrain; instantiate the named estimator.
- Accuracy is the only metric this notebook prints. Precision, recall, and
  F1 live in `get_metrics_of_models.ipynb`.

## `evaluate_loaded_dl_models.ipynb`

**Purpose.** Rebuild the Keras embedding matrix, load
`model/best_model_multi_modal` and `model/best_model_single_modal`, and
call `model.evaluate`.

**Recorded scores.**

| Checkpoint | Test acc | Subtest acc |
| --- | ---: | ---: |
| `best_model_single_modal` | 0.8635 | 0.8669 |
| `best_model_multi_modal` | 0.8735 | 0.8921 |

Those match the filenames that appear later in the metrics notebook
(`best_model_w_0.8634…`, `best_model_we_0.8734…`).

**Watch-outs.**

- The SavedModel directories are missing `variables/`. Loading will fail
  on this clone until those shards are restored.
- The notebook assigns `X_val = X_test`. That was also the validation set
  during training, so the printed test accuracy is the same split the
  checkpoint was picked on.

## `get_metrics_of_models.ipynb`

**Purpose.** One place for sklearn + Keras metrics and the comparison bar
chart used in the course write-up.

**Inputs.**

- `glove_tt.txt` (text GloVe, not the `.bin` used by the other notebooks)
- `emoji2vec_twitter.bin`
- `dataset/train_*.csv` and friends (this notebook *does* use the
  `dataset/` prefix)
- `baseline_models/{svm,dt,rf}_model.pkl` and `*_model_we.pkl` — **different
  filenames** from `baseline_models.ipynb`

**Outputs.** Printed metric cards, a matplotlib figure, and hardcoded
`svm_list` / `dt_list` / `rf_list` / `dl_list` arrays that
[experiments.md](experiments.md) transcribes.

**Watch-outs.**

- Gradient boosting is scored for accuracy and F1 in earlier cells but is
  left out of the final four-model bar chart.
- The Keras load cells point at long directory names that encode the
  scores. This clone stores equivalent graphs under
  `model/best_model_single_modal` and `model/best_model_multi_modal`.
- Chinese comments in a couple of cells (`加载GloVe词向量模型`) are from
  the original working notes and have no extra meaning.

## Recommended reading order

1. This file, then [project-overview.md](project-overview.md)
2. `data_utils.py` and [data-pipeline.md](data-pipeline.md)
3. `dl_model.py`, `attention_layer.py`, [architecture.md](architecture.md)
4. The executed output cells in `get_metrics_of_models.ipynb`
5. `examples/` if you want to press `enter` without the 2023 environment
