# Preprocessing path

Two parallel feature pipelines share a tokenizer and then diverge.

```
*.csv  ──ReadOpen──► list[list[str]]  ─┬─► mean GloVe          ──► 200-d  ── sklearn
                                       ├─► mean emoji2vec      ──► 200-d  ─┘ concat 400-d
                                       └─► Keras Tokenizer
                                            + pad_sequences
                                            + embedding matrix ──► BiLSTM + Attention
```

The runnable replay is `python3 examples/embedding_pipeline.py`.
It uses 16-d dummy vectors so you can see shapes on a laptop. The math
and the empty-tweet behaviour match `data_utils.py`.

## 1. `ReadOpen(filename, Labelfile)`

```python
sentence = " ".join(line.strip().split(","))
tokens   = [t.lower() for t in TweetTokenizer().tokenize(sentence)]
```

Consequences:

* One physical line = one tweet. A mid-line comma becomes a space.
* Every token is lower-cased, including hashtags (`#Not` → `#not`) and
  the anonymised mention token `<user>`.
* Labels are read with pandas (`header=None`) and squeezed to a 1-d
  array. The example library does the same with the stdlib.

`python3 examples/tokenize_tweets.py --text "love this, obviously #not 😒"`
prints the comma-join next to the token list.

## 2. Mean-pooled features (`ml_read_data`)

`AverageVectorPerTweet` walks each token, keeps it if
`token in glove_model.vocab`, and stores the mean. A tweet with no
in-vocab tokens becomes `zeros(200)`.

`AverageVectorPerEmoji` is the same loop against `emoji2vec_model`.
Most tokens are words, so the emoji mean is often the zero vector —
that is expected, and it is why the concatenated 400-d vector is not
"twice as informative" on every row.

`ml_read_data` then:

1. builds the 200-d word matrix `X`
2. concatenates word + emoji means into `X_emoji`
3. draws **one** permutation and applies it to both views so the
   sklearn models see aligned shuffles

The permutation uses `np.random.permutation` with no saved seed. If
you re-fit the classical models you will not bit-match the pickles.

## 3. Sequence features (`Preprocess`)

This is the deep-learning path.

1. `Tokenizer().fit_on_texts(docs)` — Keras default: lower-case,
   filter punctuation. This is **not** identical to TweetTokenizer.
   Hashtags and emoji can be chopped differently here than in the
   sklearn path. That discrepancy is historical; the examples keep
   the tweet-tokenizer view and then build a first-seen `word_index`
   so the padding step is still visible.
2. `pad_sequences(..., padding="post")`. `maxlen` is the longest
   train tweet after encoding (78 in the saved Keras summaries).
3. An embedding matrix of shape `(count, 200)` is filled as:

   * GloVe hit → copy the 200-d row
   * else run `emoji.emoji_list(word)`, keep characters that
     `emoji.is_emoji`, average their emoji2vec rows
   * else zeros

   `get_emoji2vec=False` forces the zero fallback and is how the
   **single-modal** deep model was trained. `True` is the multi-modal
   matrix (words from GloVe, emoji tokens from emoji2vec).

4. `preprocess_test` reuses the train tokenizer and the train
   `maxlen` so test / subtest rows stay 78 steps long.

## 4. What the dummy pipeline preserves

`examples/lib/embeddings.py`

| Original | Example stand-in |
| --- | --- |
| `glove.twitter.27B.200d.bin` | `DummyKeyedVectors.random_from_tokens` |
| `emoji2vec_twitter.bin` | a second dummy table on emoji tokens only |
| `AverageVectorPerTweet` | `average_vectors(docs, word_model)` |
| `AverageVectorPerEmoji` | `average_vectors(..., predicate=emoji_token_predicate)` |
| `np.concatenate(..., axis=1)` | `concatenate_modalities` |
| `pad_sequences` | `build_padded_sequences` (index 0 = pad) |
| empty tweet → `zeros(200)` | empty tweet → `zeros(dim)` |

If you change averaging (max-pool, skip zeros, etc.), do it in the
example library first and compare shapes against a few subtest rows
before touching `data_utils.py`.

## 5. Things that bite

* **Two tokenizers.** sklearn sees TweetTokenizer tokens; the Keras
  path sees `keras_preprocessing.text.Tokenizer` tokens. A hashtag
  that is one token in the first path may be `#` + `not` in the
  second, depending on filters.
* **`count` vs `word_index`.** `Preprocess` sizes the embedding
  matrix with `count=len(lines)`, not `len(tokenizer.word_index)+1`.
  That happens to be large enough on this dump (39,780 train rows,
  far fewer unique tokens) but it is not the usual Keras recipe.
* **No saved tokenizer.** Reloading `model/best_model_*` without
  re-running `Preprocess` on the same train file will not give you
  the same integer ids.
* **Gensim API.** The 2023 code uses `model.vocab` and `model[token]`,
  i.e. gensim 3.x. Gensim 4 renamed that to `key_to_index`.
