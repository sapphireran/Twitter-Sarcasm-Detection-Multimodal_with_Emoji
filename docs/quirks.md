# Quirks worth knowing before you edit the 2023 files

These are observations about the committed code and artifacts, written
so a later personal checkout does not rediscover them the hard way.

## Readers

* `ReadOpen` splits every line on commas and joins with spaces. Quoted
  CSV fields are not honored. Quotes remain on the tokens.
* `ReadOpen` returns `len(lines)` as `count`. `Preprocess` uses that as
  the first dimension of the embedding matrix. The correct Keras size
  is `len(tokenizer.word_index) + 1`.
* `ml_read_data` shuffles with an unseeded `np.random.permutation`.

## Gensim

* Membership tests use `model.vocab`, which exists in Gensim 3 and not
  in Gensim 4. Example code uses a `dict` on purpose.

## Baseline notebook fallbacks

When a pickle is missing, `baseline_models.ipynb` is supposed to train
the named estimator. Several `except` branches train an `SVC` instead:

* Decision Tree multi-modal (`dt_classifier_we`)
* Random Forest single-modal (`rf_classifier`)
* Gradient Boosting single-modal (`gbt_classifier`)

The recorded outputs were produced from successfully loaded pickles, so
the published numbers are not from those fallbacks.

## Pickles and SavedModels

* `baseline_models/` has Decision Tree and GBT only. SVM and Random
  Forest files are referenced (`svm_classifier.pkl`, `rf_classifier.pkl`,
  and also `svm_model.pkl` in the metrics notebook — two naming
  schemes).
* `model/best_model_single_modal` and `model/best_model_multi_modal`
  contain `saved_model.pb` and `keras_metadata.pb` but no `variables/`
  directory. `tf.keras.models.load_model` will not restore weights from
  this snapshot.
* The metrics notebook originally loaded directories named
  `best_model_w_<acc>_sub_<acc>` and `best_model_we_<acc>_sub_<acc>`.
  Those names encode the scores in [results.md](results.md).

## Training hyperparameters that are easy to miss

* Embedding width is 200 everywhere (GloVe Twitter 200-d).
* BiLSTM hidden size is 256 per direction, so attention sees 512-d
  states.
* Attention bias is length `steps`, so the layer is built for a fixed
  padded length (78 in the saved summaries).
* `PrepModel` uses `Adam(lr=0.001)`. Current Keras wants
  `learning_rate`.
* Dropout is 0.25 after the embedding and 0.4 after each BiLSTM.

## Path inconsistency

| Notebook | Sentence paths |
| --- | --- |
| `baseline_models.ipynb` | `train_sentence.csv` (CWD) |
| `evaluate_loaded_dl_models.ipynb` | `train_sentence.csv` (CWD) |
| `get_metrics_of_models.ipynb` | `dataset/train_sentence.csv` |

Copy or symlink the CSVs into CWD if you reopen the first two.

## Example tokenizer vs NLTK

`examples/tokenize.py` is a regex stand-in, not a bit-identical
`TweetTokenizer`. It is tested for the tokens this project cares about
(hashtags, mentions, URLs, emoji, lowercasing). Do not use it to claim
identical vocabulary statistics to the 2023 Keras `Tokenizer`.
