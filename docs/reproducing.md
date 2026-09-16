# Reproducing the 2023 run

There are two stacks on purpose.

| Goal | What to install | What to run |
| --- | --- | --- |
| Read the project, audit the CSVs, step through the math | Python 3.10+ and `pip install -r requirements-examples.txt` (NumPy) | `examples/*.py`, `python3 -m unittest discover -s tests -v` |
| Reload the June 2023 notebooks and saved Keras models | A 2023-era TensorFlow / gensim / nltk environment plus GloVe-Twitter 200d | The three `.ipynb` files, after the path fixes below |

This page documents both. It does not download GloVe or pin a full TensorFlow wheel: those choices belong to whoever is rebuilding the course VM.

## A. Documentation stack (this PR)

From the repo root:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/01_dataset_overview.py
python3 examples/02_tokenize_tweets.py
python3 examples/03_sarcasm_cues.py
python3 examples/04_tiny_embedding_pipeline.py
python3 examples/05_attention_math.py
python3 examples/06_toy_baseline.py
python3 examples/07_split_consistency.py
python3 examples/08_format_results_table.py
python3 -m unittest discover -s tests -v
```

No network, no `emoji2vec*.bin`, no GPU. Fixtures live in `examples/fixtures/`. The overview / cue / consistency scripts read `dataset/*.csv` directly.

## B. Original notebook stack

### Files the notebooks expect

| Artifact | In this git tree? | Notes |
| --- | --- | --- |
| `dataset/{train,test,subtest}_{sentence,label}.csv` | yes | Prefer these paths |
| `emoji2vec_twitter.bin` | yes (~1.3 MB) | Used by every notebook |
| `emoji2vec.bin` | yes (~2.0 MB) | Present; notebooks load the `_twitter` file |
| `glove.twitter.27B.200d.bin` | **no** | `baseline_models.ipynb`, `evaluate_loaded_dl_models.ipynb` |
| `glove_tt.txt` | **no** | `get_metrics_of_models.ipynb` (text GloVe) |
| `model/best_model_single_modal/` | partial | `keras_metadata.pb` + `saved_model.pb`; variables may be incomplete |
| `model/best_model_multi_modal/` | partial | same |
| `baseline_models/dt_classifier*.pkl` | yes | |
| `baseline_models/gbt_classifier*.pkl` | yes | |
| `baseline_models/svm_*.pkl`, `rf_*.pkl` | **no** | Referenced, not uploaded |
| NLTK `tweet` tokenizer models | no | `nltk.download` needed for `TweetTokenizer` |

Obtain GloVe-Twitter 27B 200d from the [GloVe project](https://nlp.stanford.edu/projects/glove/) (`glove.twitter.27B.zip`) and convert it to word2vec binary if you want to match `KeyedVectors.load_word2vec_format(..., binary=True)`. The `.txt` form matches the metrics notebook.

### Path inconsistencies to fix before hitting Run All

`ReadOpen` and `ml_read_data` take whatever path you pass. The notebooks disagree with each other:

| Notebook | Sentence files | GloVe file |
| --- | --- | --- |
| `baseline_models.ipynb` | `train_sentence.csv` (repo root) | `glove.twitter.27B.200d.bin` |
| `evaluate_loaded_dl_models.ipynb` | `train_sentence.csv` (repo root) | `glove.twitter.27B.200d.bin` |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` | `glove_tt.txt` (text, `binary=False`) |

Either symlink the CSVs into the repo root or edit the first cells to `dataset/…`.

### Suggested 2023-shaped pip set

These are the import names the code uses, not a lockfile. Versions around the original run (Keras 2 / TF 2.x, gensim 3.x with `.vocab`) are more likely to load the pickles and `SavedModel`s than current gensim 4 / Keras 3:

```text
tensorflow
keras-preprocessing
gensim
nltk
emoji
pandas
numpy
scikit-learn
joblib
```

Then:

```python
import nltk
nltk.download("punkt")          # sometimes pulled in transitively
# TweetTokenizer is in nltk.tokenize and does not need extra corpora
```

`attention_layer.py` notes it was tested with **Keras 2.0.6**. The saved models in `model/` were exported through TensorFlow `ModuleWrapper` layers (see the 2023 `summary()` dump). Loading them in Keras 3 will likely fail; TF 2.10–2.13 is the realistic window.

### `data_utils.py` vs current gensim

`token in model.vocab` is gensim 3. In gensim 4 use `token in model.key_to_index` or `token in model`. If you modernize that file, keep a personal branch; this archive keeps the 2023 source as the experimental record.

## C. What you should *not* expect to reproduce exactly

- SVM and RF scores, unless you find the missing pickles or accept a new seed.
- Neural training curves: the fit notebook was never uploaded.
- Bit-identical attention weights: the custom layer does not implement `get_config`, which is one reason the saved models wrap it.

If you only need the tables for a write-up, use [`results/recorded_metrics.json`](results/recorded_metrics.json) and `examples/08_format_results_table.py`. That is the honest reproduction of the *reported* experiment.
