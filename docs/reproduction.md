# Reproduction

Two reproduction stories exist.

1. **Examples (supported).** Numpy + the Python standard library. This is what `examples/` and `tests/` cover.
2. **Original notebooks (best-effort).** 2023 TensorFlow / Gensim / NLTK plus two large embedding files. Not fully automated.

## Story 1 — examples

From the repo root, on Python 3.9+ with `numpy`:

```bash
python3 examples/inspect_dataset.py
python3 examples/run_pipeline.py
python3 examples/run_attention.py
python3 examples/run_toy_classifier.py
python3 -m examples.run_all
python3 -m unittest discover -s tests -v
```

No GloVe, no TensorFlow, no NLTK, no network. Hash embeddings are deterministic, so the printed cosine values and the toy-classifier metrics are stable across machines.

`requirements.txt` lists the *original* experiment stack. You do not need it for Story 1. If you want it anyway:

```bash
python3 -m pip install -r requirements.txt
```

NLTK still needs the punkt / tweet-tokenizer tables the first time you import `ReadOpen`:

```python
import nltk
nltk.download("punkt")
```

`TweetTokenizer` itself ships with NLTK and does not need an extra corpus in recent versions.

## Story 2 — original notebooks

### Files that are not in git

| File | Used by | Notes |
| --- | --- | --- |
| `glove.twitter.27B.200d.bin` | `baseline_models.ipynb`, `evaluate_loaded_dl_models.ipynb` | Convert the Stanford Twitter GloVe 27B 200-d text file with `gensim.scripts.glove2word2vec` then `KeyedVectors.save_word2vec_format(..., binary=True)` |
| `glove_tt.txt` | `get_metrics_of_models.ipynb` | Same vectors, word2vec **text** format |
| `baseline_models/svm_*.pkl`, `rf_*.pkl` | both sklearn notebooks | Not uploaded. Re-fitting hits the buggy fallback in `baseline_models.ipynb` |
| `model/best_model_w_*`, `model/best_model_we_*` | metrics notebook | Long folder names. Use `model/best_model_{single,multi}_modal` instead |
| SavedModel variable shards | `tf.keras.models.load_model` | Only `saved_model.pb` + `keras_metadata.pb` are present |

`emoji2vec_twitter.bin` **is** in git (~1.3 MB).

### Path mismatches

| Notebook | Sentence path it opens | Actual path |
| --- | --- | --- |
| `baseline_models.ipynb` | `train_sentence.csv` | `dataset/train_sentence.csv` |
| `evaluate_loaded_dl_models.ipynb` | `train_sentence.csv` | `dataset/train_sentence.csv` |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` | correct |

Easiest fix: copy or symlink the six CSV files into the repo root before hitting Run All.

### Python / TF combo that produced the outputs

The notebooks do not pin versions. From the graph (`module_wrapper_*`, `Adam(lr=...)`, `tensorflow.python.keras.layers.embeddings.Embedding`):

- TensorFlow 2.8–2.11 era with the bundled Keras
- Gensim 3.x (`KeyedVectors.vocab`)
- `keras_preprocessing` as a standalone package
- Python 3.8 / 3.9 is the safe guess

A current `pip install tensorflow gensim` will not import `dl_model.py` unchanged.

### Suggested notebook order

1. Fix paths or symlink CSVs.
2. Place GloVe next to the notebook as the filename that notebook already uses.
3. Run `evaluate_loaded_dl_models.ipynb` first if the SavedModel loads — it is read-only.
4. Run `baseline_models.ipynb` only if you accept that missing SVM/RF pickles will train the wrong classes (or edit those cells first).
5. `get_metrics_of_models.ipynb` last; point its `load_model` cells at `model/best_model_*`.

### Seeding

`ml_read_data` shuffles with an unseeded `np.random.permutation`. Feature-to-label alignment is internally consistent, but a new sklearn fit will not match the 2023 pickles. The deep path (`ReadOpen` → `Preprocess`) does **not** shuffle, so `evaluate` can match if the tokenizer and GloVe rows match.

### Hardware

The 2023 `evaluate` traces show ~13 ms/step on 32-tweet batches (63 steps for 2,000 rows). A CPU is enough for inference. Training the Bi-LSTM on 40 k tweets wants a GPU but is not required to read this repo.

## Story 3 — just look at the data

```bash
python3 examples/inspect_dataset.py --split all
```

Prints counts, class balance, length quartiles, emoji rates, and the most common hashtags per split. This is the fastest way to see why the subtest is a different distribution.
