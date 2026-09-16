# Preprocessing

Two preprocessing stacks share a tokenizer and then diverge.

## Shared first stage: `ReadOpen`

```python
tokenizer_tweet = TweetTokenizer()
sentence = ' '.join(line.strip().split(','))
tokens = [token.lower() for token in tokenizer_tweet.tokenize(sentence)]
```

Effects that matter when you debug a prediction:

* Interior commas become spaces (legacy CSV handling).
* Mentions in this dump are already `<user>`, not `@name`.
* Hashtags stay intact (`#not` is one token).
* Emoji usually survive as their own tokens, which is what lets emoji2vec
  see them later.

`sarcasm_lib.tokenize.tokenize_tweet` is a smaller stand-in for
`TweetTokenizer`. It keeps URLs, `#hashtags`, `@mentions`, `<user>`, and
emoji atomic, and lowercases the word-like groups. It is close enough for
the documentation examples and is covered by `tests/test_tokenize.py`.

## Classical path: mean pooling

`AverageVectorPerTweet` and `AverageVectorPerEmoji` walk the token list and
average every row that exists in the corresponding `KeyedVectors` vocabulary.
Tokens that miss both tables are dropped. If a tweet contributes no rows, the
function writes a 200-d zero vector.

`ml_read_data` then:

1. builds the 200-d word matrix `X`
2. concatenates word + emoji means into `X_emoji` (400-d)
3. draws one permutation and applies it to both matrices and both label
   vectors

The permutation is why you cannot compare row `i` of `X` to line `i` of
`train_sentence.csv` after a `ml_read_data` call. The example scripts never
shuffle, so their printed tweets stay aligned with the files.

## Deep path: frozen embedding matrix

`Preprocess` fits a Keras `Tokenizer` on the already-tokenized training
lists, converts texts to integer ids, and post-pads to the longest training
tweet.

For each `word → id` it fills `embedding_matrix[id]` as follows:

1. If `word` is in GloVe-Twitter 200d, copy that row.
2. Otherwise run `emoji.emoji_list(word)` and keep characters that
   `emoji.is_emoji` accepts.
3. If any emoji remain and `get_emoji2vec=True`, write the mean emoji2vec
   row.
4. Otherwise write zeros.

Index `0` is unused padding. `count` passed into `zeros((count, 200))` is
the number of *documents*, not `len(tokenizer.word_index) + 1`. That is a
quirk of the original code: it happens to be large enough for this corpus
because there are far more tweets than unique tokens, but it is the wrong
dimension if you reuse `Preprocess` on a tiny sample. The examples do not
call `Preprocess` for that reason.

`preprocess_test` reuses the training tokenizer and the training `maxlen`
(78 in the saved models). Test tokens that were never seen at fit time
become zeros after padding; they do not grow the vocabulary.

## External vectors that are not in git

| File | Used by | In this repo? |
| --- | --- | --- |
| `glove.twitter.27B.200d.bin` / `glove_tt.txt` | Every trained model | No — too large |
| `emoji2vec_twitter.bin` | Multi-modal path | Yes (~1.3 MB) |
| `emoji2vec.bin` | Alternate emoji table | Yes (~2.0 MB) |

You can still inspect emoji geometry without GloVe. See
`examples/inspect_emoji2vec.py`, which only loads `emoji2vec_twitter.bin`.

## Randomness

`ml_read_data` shuffles with the global NumPy RNG and does not take a seed.
The original metric tables are therefore attached to the *saved* `.pkl` and
Keras directories, not to a retrained-from-scratch run. `docs/reproduction.md`
lists the checkpoints that produced those numbers.
