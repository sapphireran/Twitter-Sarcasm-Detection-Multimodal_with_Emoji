# Notebooks

Three Jupyter notebooks are the original experiment surface. They were executed in June 2023. Cells still contain outputs.

## `baseline_models.ipynb`

Purpose: fit or reload four sklearn model pairs and print accuracy on test + subtest.

Flow:

1. Load `glove.twitter.27B.200d.bin` and `emoji2vec_twitter.bin` via `KeyedVectors.load_word2vec_format(..., binary=True)`.
2. `ml_read_data` on `train_sentence.csv` / `test_sentence.csv` / `subtest_sentence.csv` **in the working directory**, not under `dataset/`.
3. For each of SVM, DT, RF, GBT:
   - `joblib.load` `baseline_models/{name}_classifier.pkl` and `{name}_classifier_we.pkl`.
   - On `FileNotFoundError`, fit a new pair and dump them.
4. Predict and `accuracy_score`.

Executed accuracies match [results.md](results.md).

Caveats:

- CSV paths omit the `dataset/` prefix.
- The `except FileNotFoundError` branches for DT / RF / GBT construct the wrong class (`SVC()` in several places). The executed run loaded pickles, so published numbers are fine. A clean-room re-run is not.
- Only DT and GBT pickles are in git today. A re-run will hit the fallback path for SVM and RF.

## `evaluate_loaded_dl_models.ipynb`

Purpose: load the two SavedModels and run `evaluate`.

Flow:

1. Same GloVe / emoji2vec load as above.
2. `ReadOpen` + `Preprocess` + `preprocess_test` on train / test / subtest (again, filenames without `dataset/`).
3. `tf.keras.models.load_model("model/best_model_multi_modal")` then `best_model_single_modal`.
4. `evaluate` on test and subtest; print `summary()`.

Executed:

| Graph | test acc | subtest acc | test loss | subtest loss |
| --- | --- | --- | --- | --- |
| multi | 0.8735 | 0.8921 | 0.3200 | 0.2852 |
| single | 0.8635 | 0.8669 | 0.3118 | 0.3056 |

`summary()` shows `module_wrapper_*` layers: the graph was saved through a TF-Keras wrapping path, not as a vanilla `Sequential` of raw LSTM objects.

## `get_metrics_of_models.ipynb`

Purpose: the full metric dump (accuracy, F1, precision, recall) plus a comparison bar chart.

Flow:

1. Loads GloVe from `glove_tt.txt` (**text** format, not the `.bin` the other notebooks use).
2. `ml_read_data` with `dataset/` prefixes (this notebook is the one that already points at the folder).
3. Loads pickles named `svm_model.pkl`, `dt_classifier.pkl`, … — **inconsistent names** versus `baseline_models.ipynb`.
4. Loads deep graphs from long folders that encode the score in the path. Those folders are not in git.
5. Prints the tables transcribed in [results.md](results.md).
6. `matplotlib` grouped bars. GBT is omitted from the figure.

Chinese comments in a few cells (`# 加载GloVe词向量模型`, `# 使用模型进行预测`) are leftover from the author's local notes. They do not change behavior.

## Suggested reading order

1. This page, then [data-pipeline.md](data-pipeline.md).
2. `examples/run_pipeline.py` so the feature math is concrete.
3. `baseline_models.ipynb` cells 3–14 if you want the sklearn loop.
4. `evaluate_loaded_dl_models.ipynb` cells 4–8 for the LSTM numbers.
5. `get_metrics_of_models.ipynb` only if you need precision / recall.

If you just want the idea, skip the notebooks and stay in `examples/`.
