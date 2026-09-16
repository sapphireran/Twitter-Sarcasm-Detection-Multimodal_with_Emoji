# Preprocessing

All of the 2023 learning code goes through [`data_utils.py`](../data_utils.py). This page is a walkthrough of that file, including the quirks that matter if you reuse it.

## 1. `ReadOpen(filename, Labelfile)`

```text
raw line
  → split on commas and re-join with spaces
  → NLTK TweetTokenizer
  → lowercase each token
  → list[list[str]]
```

Labels are `pandas.read_csv(..., header=None)` then `squeeze()` to a 1-d array.

**Comma join.** Tweets that were quoted because they contained commas lose those commas. `"So many useless classes , great to be student"` becomes tokens without a comma character. That is harmless for bag-of-vectors models and slightly destructive for anything that cares about punctuation as a feature.

**TweetTokenizer.** Keeps hashtags, emoji, and emoticons as single tokens. `😭 😭 😭` stays three tokens, not a grapheme cluster soup. `#Not` becomes `#not` after lowercasing, which is why cue counts in the docs use a case-insensitive regex.

**Return value.** `(data, labels, len(lines))`. The third item is the number of sentence *lines*, not the vocabulary size. That number is later used as `count` when allocating the embedding matrix. See the pitfall below.

## 2. Mean pooling for sklearn: `AverageVectorPerTweet` / `AverageVectorPerEmoji`

For each tweet, collect every token that exists in the keyed-vectors vocabulary and average those rows. If nothing matched, write a 200-d zero vector.

```text
tokens t1..tT
  keep { E[t] | t in vocab }
  if empty → 0_200
  else     → mean of the kept rows
```

Word and emoji tables are queried **separately**:

- `AverageVectorPerTweet` only looks up `glove_model`.
- `AverageVectorPerEmoji` only looks up `emoji2vec_model`.

A token that is an emoji will typically miss in GloVe-Twitter (or hit a rare emoji key) and hit in emoji2vec. A word will do the opposite. Tweets with no emoji therefore get a zero emoji half.

Gensim’s older API (`model.vocab` and `model[token]`) is what the 2023 file uses. Current gensim wants `token in model` / `model.key_to_index`. The lite stand-in in `examples/lite_pipeline.py` uses a plain `dict`.

## 3. `ml_read_data`

```text
ReadOpen
  X_word  = AverageVectorPerTweet(...)          # (N, 200)
  X_emo   = AverageVectorPerEmoji(...)          # (N, 200)
  X_multi = concat(X_word, X_emo)               # (N, 400)
  permute both views with the same index order
  return X_word, y, X_multi, y_multi
```

`y` and `y_multi` are the same labels after the same permutation. The extra copy exists because the notebooks treat the two views as parallel datasets.

The permutation is a **shuffle of the whole split**, including test. That does not leak train into test (each file is loaded on its own), but it does mean you cannot line up `X_test[i]` with line *i* of `test_sentence.csv` after `ml_read_data` returns. The Keras path (`Preprocess`) does **not** shuffle.

## 4. Sequence path: `Preprocess` and `preprocess_test`

Used by the BiLSTM.

1. Fit a Keras `Tokenizer` on the already-tokenized train tweets (`fit_on_texts` sees lists of tokens).
2. `texts_to_sequences` + `pad_sequences(..., padding='post')`. Train length becomes `maxlen` (78 in the saved models).
3. Build `embedding_matrix` with 200-d rows:
   - If the token is in GloVe, copy that row.
   - Else run `emoji.emoji_list` / `emoji.is_emoji` on the token, average any emoji2vec rows found, and optionally write that average (`get_emoji2vec=True` is the multi-modal setting).
   - Else write zeros.
4. `preprocess_test` reuses the train tokenizer and the train `maxlen`.

Single-modal vs multi-modal for the neural net is **not** a second input tower. It is the same architecture with a different embedding table: emoji tokens are either zero (`get_emoji2vec=False`) or emoji2vec (`True`).

## 5. Embedding-matrix size pitfall

```python
embedding_matrix = zeros((count, 200))
```

`count` is `len(lines)` (number of tweets), **not** `len(tokenizer.word_index) + 1`. `PrepModel` then does `Embedding(count, 200, weights=[embedding_matrix], ...)`.

This works only because the training set is large (39,780 tweets) relative to the tweet vocabulary. If anyone ever fits `Preprocess` on a tiny slice, `word_index` can exceed `count` and the assignment `embedding_matrix[i] = ...` will throw. The example pipeline allocates `max(index) + 1` instead and documents the difference.

Saved Keras models in `model/best_model_*` show sequence length **78** and embedding width **200**, which matches a train-fit `maxlen` from this corpus.

## 6. What the examples implement

| Original | Lite stand-in | Notes |
| --- | --- | --- |
| `ReadOpen` | `lite_pipeline.read_open` | Regex tokenizer; optional comma-join to mimic 2023 |
| `AverageVectorPerTweet` | `lite_pipeline.mean_pool` | Dict embeddings, configurable width |
| `ml_read_data` | `lite_pipeline.pooled_views` | Same  word / concat structure, no shuffle by default |
| `Preprocess` | `lite_pipeline.build_embed_table` | Allocates by vocab size |
| Attention `call` | `lite_pipeline.raffel_attention` | NumPy, same tanh / masked-softmax math |

Run `examples/02_tokenize_tweets.py` and `examples/04_tiny_embedding_pipeline.py` to see the two pooling views on a 12-tweet fixture.
