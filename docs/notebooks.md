# Notebooks

The original course work is three Jupyter notebooks plus the small Python modules they import. This page is a map, not a rewrite.

## `baseline_models.ipynb`

**Job:** fit or reload the sklearn baselines and print accuracy on `test` and `subtest`.

| Cell group | What it does |
| --- | --- |
| Imports | `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, Gensim, joblib, `ml_read_data`. |
| Vectors | Loads `glove.twitter.27B.200d.bin` and `emoji2vec_twitter.bin` from the CWD. |
| Data | `ml_read_data("train_sentence.csv", ...)` — files are expected **next to the notebook**, not under `dataset/`. |
| Per algorithm | `try: joblib.load` else fit single-modal then multi-modal and dump into `baseline_models/`. |
| Scores | `accuracy_score` only. F1 / P / R live in the metrics notebook. |

Known issues, left as they were in 2023:

- Decision-tree multi-modal cell constructs `SVC()` (copy-paste).
- Random-forest pickles are referenced but not present in this snapshot.
- SVM pickles are referenced under two names (`svm_classifier.pkl` here, `svm_model.pkl` in the metrics notebook) and neither is present.

## `evaluate_loaded_dl_models.ipynb`

**Job:** rebuild padded sequences, load the two SavedModels, call `evaluate`.

| Step | Detail |
| --- | --- |
| Vectors | Same GloVe `.bin` + `emoji2vec_twitter.bin` as the baseline notebook. |
| Data | `ReadOpen` on `train_` / `test_` / `subtest_` in the CWD, then `Preprocess` / `preprocess_test`. |
| Load | `tf.keras.models.load_model("model/best_model_multi_modal")` then `.../best_model_single_modal`. |
| Numbers | Multi-modal `0.8735` / `0.8921`; single-modal `0.8635` / `0.8669`. |
| Extra | Prints `model.summary()` (`sequential_6` / `sequential_5`, pad length 78). |

It does not compute F1. Use the metrics notebook or `sklearn.metrics` on `predict` output.

## `get_metrics_of_models.ipynb`

**Job:** the full 4-metric table and the comparison bar chart.

| Step | Detail |
| --- | --- |
| Vectors | Loads **`glove_tt.txt`** (text word2vec) plus `emoji2vec_twitter.bin`. Different filename from the other two notebooks. |
| Data | Reads from `dataset/…` (the paths this repo actually has). |
| sklearn | Reloads pickles, predicts, prints acc / F1 / recall / precision for SVM, DT, RF, GBT. |
| Deep | Loads two checkpoints whose *original* directory names encoded the scores (`best_model_w_0.8635…`, `best_model_we_0.8735…`). Those names were later copied to `model/best_model_single_modal` and `model/best_model_multi_modal`. |
| Chart | Matplotlib grouped bars for acc + F1. The PNG is stored inside the notebook. |

If you re-run this notebook today, start by pointing the deep `load_model` calls at `model/best_model_*` and the GloVe load at whichever format you actually downloaded.

## Supporting modules

| File | Imported by | Role |
| --- | --- | --- |
| `data_utils.py` | all three notebooks | Tokenize, mean-pool, build the Keras embedding matrix. |
| `dl_model.py` | training cells (not in the surviving notebooks) | `PrepModel`. |
| `attention_layer.py` | `dl_model.py`; needed as `custom_objects` on modern TF | Raffel attention. |

## Path cheat sheet

Assume the repo root is the CWD.

| Notebook | Sentence files it opens | GloVe it opens | Models it opens |
| --- | --- | --- | --- |
| `baseline_models.ipynb` | `./train_sentence.csv` (missing; use `dataset/`) | `./glove.twitter.27B.200d.bin` | `baseline_models/*.pkl` |
| `evaluate_loaded_dl_models.ipynb` | `./train_sentence.csv` (missing; use `dataset/`) | `./glove.twitter.27B.200d.bin` | `model/best_model_*` |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` | `./glove_tt.txt` | `baseline_models/*.pkl`, old `model/best_model_w_*` names |

The docs and `examples/` scripts always use `dataset/` and the two checked-in emoji2vec binaries.

## What to run instead of the notebooks

When you only want to understand the snapshot, stay on the examples (stdlib + NumPy):

```bash
python examples/explore_dataset.py
python examples/tokenize_tweets.py
python examples/inspect_emoji2vec.py --compare
python examples/attention_demo.py
python examples/cue_baseline.py
python examples/reprint_course_results.py
```

That path does not replace a full retrain. It does let you read the dataset, the embedding geometry, the attention equations, a live cue baseline, and the 2023 score table without installing TensorFlow.
