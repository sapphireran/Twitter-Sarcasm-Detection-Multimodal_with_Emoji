# Methodology

How the 2023 notebooks turn a tweet into a vector, and how the
docs/examples path differs.

## Tokenization (notebooks)

`data_utils.ReadOpen`:

1. Read a raw line from `{split}_sentence.csv`
2. `sentence = ' '.join(line.strip().split(','))` — commas become spaces
3. `nltk.TweetTokenizer().tokenize(sentence)`
4. lowercase every token

`Preprocess` then fits a Keras `Tokenizer` on those token lists, converts
to integer sequences, and `pad_sequences(..., padding='post')`. The
padded width is the longest train tweet (78 in the saved models).

Test/subtest reuse the train tokenizer via `preprocess_test`.

## Tokenization (examples)

`sarcasm_toolkit.tokenize.tokenize_tweet` is a regex tokenizer that keeps
`#not`, `<user>`, URLs, and emoji clusters as single tokens. It is close
enough for cue counting. It is **not** claimed to match NLTK. Do not mix
the two when reproducing notebook metrics.

## Word vectors

Notebooks load Stanford Twitter GloVe 27B 200d, converted to word2vec
binary (`glove.twitter.27B.200d.bin` or `glove_tt.txt`). That file is
not in git.

* Classical models: `AverageVectorPerTweet` — mean of in-vocab GloVe
  rows; missing-all-tokens → 200 zeros
* Deep model: rows of the Keras embedding matrix are GloVe vectors
  indexed by the Keras tokenizer

## Emoji vectors

`emoji2vec_twitter.bin` (and `emoji2vec.bin`) **are** in the repo.

* Classical multi-modal: `AverageVectorPerEmoji` then
  `concat(word_mean, emoji_mean)` → 400-d
* Deep multi-modal: if a Keras token is not in GloVe,
  `emoji.emoji_list` / `emoji.is_emoji` extracts emoji characters and
  averages their emoji2vec rows into that embedding slot
  (`get_emoji2vec=True`). Single-modal sets those slots to zero.

That is why the two saved BiLSTMs share a graph but not an embedding
table.

## Training notes from the notebooks

* Adam, learning rate `0.001`, binary cross-entropy, accuracy metric
  (`dl_model.PrepModel`)
* Dropout 0.25 after embedding, 0.4 after each BiLSTM
* Evaluation used the Keras `evaluate` path on test (63 batches) and
  subtest (9 batches) in `evaluate_loaded_dl_models.ipynb`
* `ml_read_data` shuffles with `np.random.permutation` every call — if
  you reload classical models, keep the same shuffle only if you also
  persist indices (the notebooks do not). Saved pickles were evaluated
  in the same session they were trained.

## Metrics

The report quotes accuracy, F1, precision, and recall from
`sklearn.metrics` with default binary (positive = sarcastic = 1).
`sarcasm_toolkit.metrics.binary_metrics` matches that definition
(undefined precision/recall become 0 when the denominator is 0).

## What the examples deliberately skip

* No GloVe download
* No TensorFlow fit/evaluate
* No live Twitter/X API
* No attempt to bitwise-reproduce the pickled sklearn models

They implement the *scientific* counterpart: dataset leakage, a lexicon
floor, and the attention formula.
