# Preprocessing

All 2023 paths start in `data_utils.py`.

## `ReadOpen`

```text
raw line → strip → split on commas → join with spaces
         → nltk.TweetTokenizer → lowercase
```

The archive lab’s `ccs2lab.tokenize.tokenize_tweet` follows the same
CSV normalization and keeps hashtags, mentions, URLs, emoji, and
multi-digit numbers together. It is not a guaranteed NLTK clone.

## Classical (sklearn) features

`ml_read_data` builds two matrices with the same row permutation:

1. **W.** `AverageVectorPerTweet`: mean of in-vocabulary GloVe vectors.
   Empty tweets become a 200d zero.
2. **WE.** concatenate W with `AverageVectorPerEmoji` (mean of
   in-vocabulary emoji2vec rows, or 200d zero).

Both helpers skip OOV tokens rather than using an UNK vector. A tweet
made of unknown slang therefore collapses to zeros on that channel.

`ml_read_data` then applies `np.random.permutation` **independently
per call** with no seed. Train and test stay internally aligned
(same indices for X and y), but you cannot compare row *i* of a
feature matrix back to line *i* of the CSV after a call. The archive
lab never shuffles unless a script says so.

## LSTM features

`Preprocess` fits a Keras `Tokenizer` on the already-tokenized train
docs, pads with `padding='post'`, and builds a `(count, 200)`
embedding matrix. `count` here is **the number of training lines**,
not `vocab_size + 1`. That only works because the training set
(39,780) is larger than the observed vocabulary. Index 0 is unused
padding.

For each `word, i` in `tokenizer.word_index`:

1. If `word` is in GloVe, copy that row.
2. Else collect emoji codepoints from `emoji.emoji_list(word)`.
3. If those emoji hit emoji2vec and `get_emoji2vec=True`, write the
   mean emoji vector.
4. Else write zeros.

`preprocess_test` reuses the train tokenizer and the train pad
length. The saved 2023 models report width **78**.

Single-modal vs multi-modal for the LSTM is the `get_emoji2vec` flag,
not a second input tower. Both saved models have the same
`summary()` shape; the difference is which embedding rows were
filled before `trainable=False`.

## Embeddings that are actually in the repo

| File | Header | Used by |
| --- | --- | --- |
| `emoji2vec_twitter.bin` | `1661 200` | 2023 WE path |
| `emoji2vec.bin` | `1661 300` | original Eisner table, unused by the notebooks |
| `glove.twitter.27B.200d.bin` / `glove_tt.txt` | **missing** | every W path |

`examples/03_emoji2vec_probe.py` reads the two checked-in binaries
with `ccs2lab.w2v` (no gensim).
