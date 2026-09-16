# Reproduction

There are two ways to work with this repository.

1. **Docs and examples** (this PR). No GloVe, no TensorFlow. Python 3.10+
   and NumPy are enough.
2. **Original notebooks.** You need the 2023 stack: TensorFlow / Keras,
   Gensim `KeyedVectors`, NLTK `TweetTokenizer`, the `emoji` package, and
   the Twitter GloVe 200d table that is not in git.

## Lightweight path

From the repository root:

```bash
python3 -m pip install -r requirements-examples.txt
python3 -m unittest discover -s tests -v
python3 examples/inspect_dataset.py
python3 examples/tokenize_tweets.py
python3 examples/heuristic_baseline.py
python3 examples/attention_walkthrough.py
python3 examples/predict_cli.py "I love walking to school #not"
python3 examples/inspect_emoji2vec.py --neighbors "😂"
python3 examples/inspect_emoji2vec.py --report
```

`inspect_emoji2vec.py` is the only example that needs Gensim. It loads
`emoji2vec_twitter.bin` from the repo root. The other scripts import
`sarcasm_lib` only.

## Original notebooks

Place GloVe next to the notebooks. The cells disagree about the filename:

| Notebook | Path they load |
| --- | --- |
| `baseline_models.ipynb` | `glove.twitter.27B.200d.bin` (word2vec binary) |
| `evaluate_loaded_dl_models.ipynb` | `glove.twitter.27B.200d.bin` |
| `get_metrics_of_models.ipynb` | `glove_tt.txt` (plain text) |

Both are conversions of [GloVe Twitter 27B, 200d](https://nlp.stanford.edu/projects/glove/).
`emoji2vec_twitter.bin` is already in the tree.

The baseline notebook also expects tweet files in the *current working
directory* (`train_sentence.csv`, …). Either copy `dataset/*.csv` into the
notebook folder or change those strings to `dataset/...`. The metrics
notebook already uses the `dataset/` prefix.

Classical pickles:

* Present: `baseline_models/dt_classifier*.pkl`, `gbt_classifier*.pkl`
* Missing from git: SVM and Random Forest joblib files. The notebook will
  try to fit them if GloVe is available. Fitting SVM on 39,780 × 400
  features is slow.

Deep checkpoints:

* `model/best_model_single_modal/`
* `model/best_model_multi_modal/`

`evaluate_loaded_dl_models.ipynb` calls `tf.keras.models.load_model` on those
directories. Custom `Attention` objects may need
`custom_objects={"Attention": Attention}` on current TensorFlow. The 2023
run loaded them without that argument.

`PrepModel` uses `Adam(lr=lrate)`. Recent Keras renamed that argument to
`learning_rate`. If you rebuild the graph instead of loading a SavedModel,
change that keyword.

## What is not reproduced here

* Training logs, epoch counts, and the validation split used to pick
  `best_model_*` were not checked in.
* `ml_read_data` shuffles with an unseeded NumPy permutation, so a fresh
  classical fit will not match the published row order.
* The published metrics should be treated as properties of the shipped
  checkpoints. See `docs/results.md`.

## Tests as a regression net

`tests/` loads the real CSV files and checks:

* split sizes (39,780 / 2,000 / 278)
* test-set balance (1,000 / 1,000)
* `#not` is the most common training hashtag
* the heuristic beats the majority class on subtest
* Raffel attention focuses on the high-scoring token in a toy sequence

If you change `dataset/` or the marker list, those tests will fail on
purpose.
