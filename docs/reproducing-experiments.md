# Reproducing the 2023 experiments

This is a personal course archive, not a locked experiment runner. Several artifacts that the notebooks expect are missing or only partly uploaded. Read this page before you spend time on a full TensorFlow install.

## File checklist

| Artifact | Expected by | In this checkout? |
| --- | --- | --- |
| `dataset/{train,test,subtest}_{sentence,label}.csv` | all notebooks, all examples | yes |
| `emoji2vec_twitter.bin` | classical + deep multimodal | yes (~1.3 MB) |
| `emoji2vec.bin` | unused by the notebooks (alternate dump) | yes (~2.0 MB) |
| `glove.twitter.27B.200d.bin` | `baseline_models.ipynb`, `evaluate_loaded_dl_models.ipynb` | **no** (gitignored) |
| `glove_tt.txt` | `get_metrics_of_models.ipynb` | **no** |
| `baseline_models/svm_classifier.pkl` and `*_we.pkl` | baseline notebook | **no** |
| `baseline_models/rf_classifier.pkl` and `*_we.pkl` | baseline notebook | **no** |
| `baseline_models/dt_classifier.pkl` and `*_we.pkl` | baseline notebook | yes |
| `baseline_models/gbt_classifier.pkl` and `*_we.pkl` | baseline notebook | yes |
| `baseline_models/svm_model.pkl` (different name) | metrics notebook | **no** |
| `model/best_model_{single,multi}_modal/` | evaluate / metrics notebooks | **partial** (only `keras_metadata.pb` is present; weight shards are not) |

If a cell says “Loaded … from files successfully,” it found the pickle. If it says `FileNotFoundError` and starts fitting, that is expected for the missing SVM/RF dumps — and fitting SVM on 39k × 200-d vectors is slow.

## Path inconsistencies

The notebooks disagree about where the CSVs live:

| Notebook | Paths used |
| --- | --- |
| `baseline_models.ipynb` | `train_sentence.csv` in the **repo root** |
| `evaluate_loaded_dl_models.ipynb` | same root-relative names |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` |

From the repo root:

```bash
ln -s dataset/train_sentence.csv
ln -s dataset/train_label.csv
ln -s dataset/test_sentence.csv
ln -s dataset/test_label.csv
ln -s dataset/subtest_sentence.csv
ln -s dataset/subtest_label.csv
```

The example scripts always read `dataset/` and do not need the symlinks.

## Suggested notebook order

1. Obtain Twitter GloVe 200-d and convert it to word2vec binary if you only have the official `.txt`. Gensim’s `KeyedVectors.load_word2vec_format(..., binary=True)` is what the notebooks call.
2. `baseline_models.ipynb` — trains or loads the four sklearn pairs, prints accuracy.
3. `evaluate_loaded_dl_models.ipynb` — only useful if you restore a complete SavedModel.
4. `get_metrics_of_models.ipynb` — the wide metric dump and the comparison plot. It expects yet another set of pickle names (`svm_model.pkl` vs `svm_classifier.pkl`). Align the filenames before you run it.

## Environment

`requirements.txt` lists the families used in 2023, with upper bounds that avoid TF 2.16+ Keras 3 breakage. Even then:

- `tensorflow.python.keras.layers.embeddings.Embedding` (imported in `dl_model.py`) is an internal path and may disappear.
- `Adam(lr=lrate)` is the old kwarg; current Keras wants `learning_rate`.
- `model.vocab` on a Gensim 4 `KeyedVectors` is `model.key_to_index`. `data_utils.py` still uses `.vocab` and will need a one-line shim on Gensim ≥ 4.
- `Attention` should be passed as `custom_objects` when calling `tf.keras.models.load_model`.

A faithful replay is easiest on a Python 3.8–3.10 environment with TensorFlow 2.8–2.11 and Gensim 3.8. A modern rewrite should port `data_utils.py` to Gensim 4 and Keras 3 instead of fighting the 2023 imports.

## What you can reproduce *without* GloVe or TensorFlow

Everything under `examples/` and `tests/`:

```bash
python3 examples/inspect_dataset.py
python3 examples/tokenize_demo.py --limit 8
python3 examples/emoji_cooccurrence.py
python3 examples/heuristic_baseline.py
python3 examples/embedding_average_demo.py
python3 examples/attention_walkthrough.py
python3 -m unittest discover -s tests -v
```

Those scripts recompute dataset statistics, a hashtag-aware tokenizer, emoji/label co-occurrence, a leakage-aware heuristic baseline, a toy averaged-embedding demo, and a NumPy clone of the attention equations.

## Randomness

`ml_read_data` shuffles with `np.random.permutation` and no seed. Retraining sklearn models will not bit-match the pickled ones even if the rest of the stack is identical. The deep model’s training seed is not recorded.

## Ethics / data use

Do not re-hydrate `<user>` tokens, scrape the original authors, or push these tweets into a public demo that invites live prediction on other people’s posts. The allowed use is local, personal, academic inspection of a 2023 course dataset.
