# Reproduction notes

Two tracks:

1. **This clone, today.** Dataset reports, tokenizer demo, attention walkthrough,
   lexical baseline. Needs Python 3.10+ and NumPy.
2. **The 2023 notebooks.** GloVe Twitter 200-d, NLTK, TensorFlow 2 / Keras,
   gensim `KeyedVectors`, and the missing pickle / SavedModel weights.

## Track 1 — examples that run here

From the repository root:

```bash
python3 -m examples.dataset_report --markdown
python3 -m examples.emoji_signals --split test
python3 -m examples.tweet_tokenizer_demo
python3 -m examples.attention_demo
python3 -m examples.metrics_table --metric accuracy
python3 -m examples.lexical_baseline_demo --train-limit 8000
python3 -m unittest discover -s tests
```

NumPy is the only third-party import. The original `data_utils.py` stack
(NLTK, gensim, Keras, emoji, pandas) is not imported.

## Track 2 — original course stack

Approximate pip set from the notebooks:

```text
tensorflow>=2.8
keras-preprocessing
gensim
nltk
emoji
pandas
scikit-learn
joblib
matplotlib
```

Then:

```python
nltk.download("punkt")  # TweetTokenizer itself is in nltk.tokenize
```

Place `glove.twitter.27B.200d.bin` next to the notebooks. The notebooks
sometimes load `glove_tt.txt` (plain word2vec text) and sometimes the `.bin`.
`emoji2vec_twitter.bin` is already in the repo root.

Training data paths are inconsistent across notebooks:

| Notebook | Sentence path |
| --- | --- |
| `baseline_models.ipynb` | `train_sentence.csv` in the cwd |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` |
| `evaluate_loaded_dl_models.ipynb` | `train_sentence.csv` in the cwd |

Copy or symlink the CSVs if you run a notebook from a different working
directory.

## Missing artifacts

| Path | Status | Effect |
| --- | --- | --- |
| `glove.twitter.27B.200d.bin` | absent | cannot build embedding matrices |
| `baseline_models/*.pkl` | empty directory | sklearn notebooks fall through to retraining |
| `model/best_model_single_modal/` | metadata only | `tf.keras.models.load_model` will fail |
| `model/best_model_multi_modal/` | metadata only | same |
| `model/best_model_w_*` / `best_model_we_*` | absent | names used in `get_metrics_of_models.ipynb` |

Re-training the LSTM from `dl_model.PrepModel` is the realistic way to recover
deep scores. Budget a GPU or a long CPU run; two stacked 256-unit Bi-LSTMs on
~40k padded sequences is not a laptop-minute job.

## Known sharp edges in the 2023 code

1. **`dl_model.py` imports `Embedding` twice**, once from
   `tensorflow.keras.layers` and once from
   `tensorflow.python.keras.layers.embeddings`. The second import wins. This
   is leftover TF 2.8 / 2.10 mixing.
2. **Adam `lr=` is deprecated** in current Keras. Use `learning_rate`.
3. **`gensim` `.vocab`** is gone in gensim 4. Use `key_in_model` / `key_to_index`.
4. **`baseline_models.ipynb` except-blocks** construct the wrong estimator for
   several WE models (`SVC()` instead of the named classifier). Harmless if
   pickles exist; harmful if you retrain from scratch.
5. **`ml_read_data` shuffles train and test with `np.random.permutation`.**
   There is no seed. Feature order is randomized every load, so you cannot
   bitwise-compare sklearn outputs across runs unless you pin the seed.
6. **Attention bias is length-shaped.** A model saved at `maxlen=78` cannot
   be applied to a different pad length without rebuilding the layer.

## Suggested modern re-run

If you rebuild this project for yourself later:

1. Keep the CSV layout and the W vs WE split. That comparison is the point.
2. Seed every shuffle.
3. Report test *and* subtest, not just test. Emoji fusion is easy to hide on
   the full test set because most tweets have no emoji.
4. Log precision and recall, not only accuracy. The 2023 WE LSTM traded recall
   for precision on the official test set.
5. Freeze a `requirements.txt` with gensim and TensorFlow versions that still
   accept `.vocab` and `Adam(lr=...)`, or patch those two calls.

The example library is already written against current NumPy and does not
inherit those version pins.
