# Notebooks

Three Jupyter notebooks are the original coursework interface. They
are left as-is. This page is a map so you do not have to execute
them to know what they contain.

## `baseline_models.ipynb`

**Purpose.** Load GloVe + emoji2vec, call `ml_read_data` on train /
test / subtest, then load-or-fit four sklearn pairs.

**Cells.**

| Cell | What it does |
| --- | --- |
| 0 | Imports (`SVC`, trees, `KeyedVectors`, `joblib`, `ml_read_data`) |
| 1 | Loads `glove.twitter.27B.200d.bin` and `emoji2vec_twitter.bin` |
| 2 | Builds `X_*` (200-d) and `X_emoji_*` (400-d) |
| 3–5 | SVM section; saved acc: 0.769 / 0.763 / 0.813 / 0.824 |
| 6–8 | Decision tree; 0.727 / 0.730 / 0.777 / 0.799 |
| 9–11 | Random forest; 0.815 / 0.818 / 0.806 / 0.853 |
| 12–14 | Gradient boosting; 0.746 / 0.748 / 0.795 / 0.795 |

**Paths.** Bare `train_sentence.csv` (not `dataset/train_sentence.csv`).
Run it with those files in the working directory, or edit the strings.

**Load vs fit.** Each section `joblib.load`s two pickles and only
trains on `FileNotFoundError`. The RF/GBT/DT fallbacks construct
the wrong class in some branches — see
[reproduction.md](reproduction.md).

## `evaluate_loaded_dl_models.ipynb`

**Purpose.** Rebuild padded sequences with `Preprocess` /
`preprocess_test`, then `tf.keras.models.load_model` on the two
SavedModel directories.

**Saved outputs.**

- Multi-modal: test acc 0.8735, subtest acc 0.8921, test loss 0.3200.
- Single-modal: test acc 0.8635, subtest acc 0.8669, test loss 0.3118.
- Both `summary()` printouts show `sequential_*` with two
  `Bidirectional` layers (512-d) and 2,510,848 trainable params.

**Will not run today** unless you restore `model/*/variables/`.

## `get_metrics_of_models.ipynb`

**Purpose.** The long-form metrics notebook: sklearn accuracy + F1
+ precision + recall, then the same for both LSTMs, then a bar
chart comparing W / WE × full / sub.

**Inconsistencies to watch.**

- GloVe is loaded from `glove_tt.txt` (`binary=False`) in cell 2,
  not from the `.bin` used by the other notebook.
- Dataset paths **do** use the `dataset/` prefix.
- Early cells load `baseline_models/svm_model.pkl` (different
  filename than `svm_classifier.pkl` in the training notebook).
- Later cells hard-code `svm_list` / `dt_list` / `rf_list` /
  `dl_list` so the plot can be redrawn without rerunning inference.
- One markdown header says `Bi-LSTM+ATT`. That is `PrepModel`.

The tables in [results.md](results.md) are taken from this
notebook’s stdout and from those hardcoded lists.

## What is not a notebook

- `dl_model.py` — graph only, no fit loop.
- `attention_layer.py` — layer class, no demo.
- `data_utils.py` — IO + embedding helpers.

The `examples/` scripts cover those three files without Jupyter.
