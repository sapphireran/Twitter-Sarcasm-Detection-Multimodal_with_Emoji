# Reproduction notes

Two different "run this repo" stories:

1. **Docs / examples (this snapshot, today).** Python 3.10+ and NumPy.
   No GloVe, no TensorFlow, no NLTK. This is what the extra files in
   `docs/` and `examples/` are for.
2. **2023 notebooks (original training machine).** Keras 2, gensim 3,
   the Twitter GloVe binary, and the missing sklearn pickles.

## Lightweight path (supported here)

```bash
python3 -m unittest tests.test_examples
python3 examples/explore_dataset.py
python3 examples/tokenize_tweets.py --split subtest --n 4
python3 examples/sarcasm_cues.py --split test
python3 examples/attention_numpy.py
python3 examples/embedding_pipeline.py --split subtest --n 6
python3 examples/report_results.py --all
```

These commands read `dataset/*.csv` and the recorded metrics module.
They do not train anything and they do not touch `emoji2vec_*.bin`.

## What is actually in the GitHub snapshot

Present:

* all six CSV files
* `attention_layer.py`, `data_utils.py`, `dl_model.py`
* the three course notebooks
* `emoji2vec.bin`, `emoji2vec_twitter.bin`
* `baseline_models/dt_classifier{,_we}.pkl`
* `baseline_models/gbt_classifier{,_we}.pkl`
* Keras `saved_model.pb` + `keras_metadata.pb` for both DL runs

Missing or incomplete:

| Artefact | Why it matters |
| --- | --- |
| `glove.twitter.27B.200d.bin` / `glove_tt.txt` | every notebook's first real cell |
| `baseline_models/svm_classifier{,_we}.pkl` | SVM cells will re-fit (slow) or look for a different filename |
| `baseline_models/rf_classifier{,_we}.pkl` | same for random forest |
| `model/best_model_*/variables/` | Keras cannot restore weights from proto files alone |
| NLTK `punkt` / tweet tokenizer models | `ReadOpen` imports `nltk.TweetTokenizer` |
| pinned `requirements.txt` | the 2023 environment was never frozen |

The metrics notebook also looks for `svm_model.pkl` (different stem
than `svm_classifier.pkl` in `baseline_models.ipynb`). That mismatch
is original.

## Original notebook environment (best-effort)

The notebooks were last executed on 5–6 June 2023. Imports imply:

```
tensorflow~=2.x          # mixed tensorflow.keras and tensorflow.python.keras
keras-preprocessing
gensim~=3.8              # .vocab + keyedvectors[token]
nltk                     # TweetTokenizer
emoji                    # emoji_list / is_emoji
scikit-learn
pandas
numpy
joblib
matplotlib               # final bar chart in get_metrics_of_models.ipynb
```

Suggested reconstruction, if you want to re-train rather than read
the recorded tables:

1. Download [GloVe Twitter 27B](https://nlp.stanford.edu/projects/glove/)
   200-d and convert it to word2vec binary (`glove.twitter.27B.200d.bin`)
   or point the metrics notebook at the `.txt` it already mentions
   (`glove_tt.txt`).
2. `pip install` the stack above on *Python 3.8/3.9*. 3.12 will fight
   you on `tensorflow.python.keras` and gensim 3.
3. `nltk.download` whatever `TweetTokenizer` needs on a clean box.
4. Run `baseline_models.ipynb` first if you want new pickles. Watch
   the except-blocks: several "train the tree" cells construct `SVC()`
   for the W model when the pickle is missing.
5. Training the BiLSTM needs a GPU you are willing to leave on for
   a while (39,780 rows, two 256-d bidirectional LSTMs, `maxlen=78`).
   `PrepModel` does not set a seed.

## Reloading the saved deep models

`evaluate_loaded_dl_models.ipynb` calls
`tf.keras.models.load_model("model/best_model_multi_modal")`.
That only works if the `variables/` directory is next to
`saved_model.pb`. It is not in this snapshot, so the cell will fail
even if you have TensorFlow. The accuracies printed in the notebook
outputs (0.8735 test / 0.8921 subtest for WE) are the record of a
run that already happened.

Custom objects: the attention layer lives in `attention_layer.py` as
class `Attention`. A modern `load_model` will need
`custom_objects={"Attention": Attention}` plus whatever
`ModuleWrapper` the mixed Keras imports created.

## Seeds, shuffles, and why you will not bit-match

* `ml_read_data` shuffles with `np.random.permutation` and no seed.
* `PrepModel` does not set `tf.random.set_seed`.
* Dropout is on at train time (0.25 / 0.4 / 0.4).
* The Keras tokenizer and NLTK tokenizer are not the same function.

Re-training should be compared on the **four-cell W/WE ×
test/subtest grid**, not on exact pickle bytes.

## Data-only checks you can do offline

These do not need the missing binaries:

```bash
python3 -m unittest tests.test_examples
python3 examples/explore_dataset.py --json
python3 examples/sarcasm_cues.py --split test --json
```

`tests.test_examples.DatasetTests` asserts the three official sizes
and that the test split is still balanced 1000/1000. If a future
edit to the CSVs breaks alignment, that test is the tripwire.
