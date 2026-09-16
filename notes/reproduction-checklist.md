# Reproduction checklist

Personal list of what a future me needs to rerun the June 2023 tables. This checkout is not self-contained.

## Present in git

- [x] Train / test / subtest sentence and label CSVs
- [x] `data_utils.py`, `dl_model.py`, `attention_layer.py`
- [x] The three notebooks with printed metrics
- [x] `emoji2vec.bin` and `emoji2vec_twitter.bin`
- [x] Decision-tree and gradient-boosting pickles (W and WE)
- [x] Keras SavedModel *graphs* for `best_model_{single,multi}_modal` (`saved_model.pb`, `keras_metadata.pb`)
- [x] These notes

## Missing or incomplete

- [ ] **GloVe Twitter 27B 200-d.** Not in the repo (too big). Notebooks want either `glove.twitter.27B.200d.bin` or `glove_tt.txt`. Same vectors, different format. Get them from the Stanford GloVe page and convert with gensim if you only have the official `.txt`.
- [ ] **Keras weight shards.** Both `model/best_model_*` folders lack `variables/variables.data-*` and `variables/variables.index`. `tf.keras.models.load_model` will fail. The 2023 metrics notebook loaded differently named directories that are also absent.
- [ ] **SVM pickles.** `baseline_models/svm_classifier.pkl`, `svm_classifier_we.pkl`, and/or `svm_model.pkl`, `svm_model_we.pkl`.
- [ ] **Random forest pickles.** `rf_classifier.pkl`, `rf_classifier_we.pkl` (and the `rf_model*` names the metrics notebook used).
- [ ] **Training script.** No committed `fit` loop, callbacks, or seed.
- [ ] **NLTK tokenizer data.** `TweetTokenizer` is in the NLTK package; some environments still want `nltk.download` bits. Check before assuming `ReadOpen` works.

## Python stack I remember

Not pinned. Reconstructing from imports and the TF 2 `module_wrapper` summaries:

- Python 3.x (3.8/3.9 likely)
- `tensorflow` 2.x with `tensorflow.keras` and leftover `tensorflow.python.keras` imports
- `keras_preprocessing` (Tokenizer, pad_sequences) — separate from tf.keras in that era
- `gensim` KeyedVectors
- `nltk` TweetTokenizer
- `emoji` (the modern `emoji_list` / `is_emoji` API; an old `emoji` package will break `Preprocess`)
- `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`

`dl_model.py` still says `Adam(lr=lrate)`. Current Keras wants `learning_rate`. That line will warn or error depending on version.

## Commands I would run, in order

Do not treat this as a CI script. It is the order that matches the notebooks.

```bash
# 1. put GloVe next to the notebooks
#    glove.twitter.27B.200d.bin   or   glove_tt.txt

# 2. confirm emoji2vec
ls emoji2vec_twitter.bin

# 3. baselines (will retrain if pickles are missing — see warning below)
#    jupyter: baseline_models.ipynb
#    data paths in that notebook are bare filenames; either cwd=dataset/
#    or edit them to dataset/train_sentence.csv etc.

# 4. metrics
#    jupyter: get_metrics_of_models.ipynb
#    this one already prefixes dataset/
```

**Warning.** If you run `baseline_models.ipynb` without the original pickles, three except-blocks fit `SVC()` under tree / forest / boosting names. Fix those cells first or you will overwrite the story.

Deep-model eval (`evaluate_loaded_dl_models.ipynb`) cannot succeed until the `variables/` directories exist or you retrain with `PrepModel` and `model.save(...)`.

## Path inconsistencies

| Notebook | GloVe | Data prefix | Sklearn names | Keras names |
| --- | --- | --- | --- | --- |
| `baseline_models.ipynb` | `glove.twitter.27B.200d.bin` | none (cwd files) | `*_classifier.pkl` | — |
| `evaluate_loaded_dl_models.ipynb` | same binary | none | — | `model/best_model_{single,multi}_modal` |
| `get_metrics_of_models.ipynb` | `glove_tt.txt` | `dataset/` | mix of `*_model.pkl` and `*_classifier.pkl` | `best_model_w_*` / `best_model_we_*` |

I am not cleaning that in this notes-only pass. Align them before a serious rerun.

## What "reproduced" would mean

Minimum I would accept:

1. GloVe + emoji2vec load, `ReadOpen` returns 39780 / 2000 / 278.
2. `Preprocess` builds a 78-wide padded train matrix (or whatever `maxlen` falls out — if it is not 78, the committed attention bias shape will not match old weights anyway).
3. Retrained WE BiLSTM beats retrained W on both slices in the same direction as 2023 (+~1 acc full test, +~2–3 acc subtest). Absolute numbers may move.
4. Forest still beats SVM / tree / boosting on full-test acc.

I would *not* require bit-identical 0.8735. No seed, incomplete artifacts, floating TF versions.

## Local files I should not commit even if I find them again

- Full GloVe (hundreds of MB to a few GB)
- `variables/` shards if they contain nothing secret — actually those *should* have been committed if I wanted the repo to evaluate. Size is the only reason I might still skip them.
- Extra snapshot directories named with raw floats

No API keys, no scraped user ids beyond what is already in the CSVs. The sentences themselves are public-looking tweets; I am not adding more raw scrapes in this cleanup.
