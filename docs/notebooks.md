# Notebooks

Three Jupyter notebooks were uploaded with the 2023 project. None of
them train the BiLSTM from scratch in this snapshot; `dl_model.PrepModel`
is the constructor that a missing training notebook would have called.

## `baseline_models.ipynb`

1. Load GloVe Twitter 200-d (binary) and `emoji2vec_twitter.bin`.
2. Call `ml_read_data` on train / test / subtest (paths relative to CWD).
3. For SVM, Decision Tree, Random Forest, Gradient Boosting:
   * try to `joblib.load` `baseline_models/<name>.pkl` and
     `baseline_models/<name>_we.pkl`
   * otherwise fit `sklearn` estimators and dump them
4. Print accuracy on test and subtest, single-modal and multi-modal.

Recorded accuracies are copied into [results.md](results.md).

## `evaluate_loaded_dl_models.ipynb`

1. Same embedding load as the baseline notebook.
2. `ReadOpen` + `Preprocess` + `preprocess_test`.
3. `tf.keras.models.load_model("model/best_model_multi_modal")` and
   `model/best_model_single_modal`.
4. `model.evaluate` on test and subtest; print `summary()`.

Recorded:

* multi-modal test acc 0.8735, subtest 0.8921
* single-modal test acc 0.8635, subtest 0.8669
* both summaries show 78-step inputs and 2,510,848 parameters

## `get_metrics_of_models.ipynb`

The long-form scoring notebook. It:

* loads GloVe from `glove_tt.txt` (text, `binary=False`)
* scores the sklearn pickles with accuracy, F1, precision, recall
* reloads the deep models under their original
  `best_model_w_*` / `best_model_we_*` directory names
* draws a grouped bar chart of accuracies and F1 scores

The last code cell hard-codes the four-column lists so the plot can be
rebuilt without rerunning inference:

```python
svm_list = [[0.769, 0.763, 0.813, 0.824], ...]
dt_list  = [[0.7265, 0.7295, 0.777, 0.799], ...]
rf_list  = [[0.8145, 0.818, 0.806, 0.853], ...]
dl_list  = [[0.8635, 0.8735, 0.867, 0.892], ...]
```

(Values rounded here; the notebook keeps full floats.)

## Missing training notebook

Nothing in the dump calls `PrepModel(...).fit(...)`. To retrain the
deep model you would:

1. Build features with `Preprocess` / `preprocess_test`.
2. `model = PrepModel(count, embedding_matrix, l)`.
3. Fit on `padded_docs_train` with a binary label vector.
4. `model.save(...)`.

That loop is intentionally not reinvented in `examples/`; the toy
pipeline uses logistic regression so it can run without TensorFlow.
