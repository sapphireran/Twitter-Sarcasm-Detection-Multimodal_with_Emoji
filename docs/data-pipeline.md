# Data pipeline

Everything that turns a raw tweet line into a model feature lives in `data_utils.py`. This page walks that file in order and records the on-disk splits.

## Files on disk

```
dataset/
  train_sentence.csv      39,780 tweets
  train_label.csv         39,780 labels, no header
  test_sentence.csv        2,000 tweets
  test_label.csv
  subtest_sentence.csv       278 tweets
  subtest_label.csv
```

There is no CSV header and no tweet id. Row \(i\) in `*_sentence.csv` matches row \(i\) in `*_label.csv`.

Labels:

| Split | n | label 0 | label 1 | P(sarcastic) |
| --- | ---: | ---: | ---: | ---: |
| train | 39,780 | 21,292 | 18,488 | 0.465 |
| test | 2,000 | 1,000 | 1,000 | 0.500 |
| subtest | 278 | 106 | 172 | 0.619 |

The subtest is **not** a random subsample of test. It is a small, sarcasm-skewed, emoji-rich hold-out. Treat it as a stress test for the emoji channel, not as a second i.i.d. test set.

`examples/inspect_dataset.py` measures the shift on the checked-in CSVs:

| Split | emoji rate | hashtag rate | `#not` / `#sarcasm` / `#sarcastictweet` / `#yeahright` |
| --- | ---: | ---: | ---: |
| train | 13.7% | 21.3% | 9.2% |
| test | 13.8% | 46.6% | 30.5% |
| subtest | **99.3%** | 55.8% | 47.8% |

The subtest is almost entirely emoji-bearing. That is why Fusion A and Fusion B move this slice and barely move the balanced test set.

## Stage 1 — `ReadOpen(filename, Labelfile)`

```python
sentence = ' '.join(line.strip().split(','))
tokens = [tok.lower() for tok in TweetTokenizer().tokenize(sentence)]
```

Why split on commas and re-join? The sentence files sometimes look like CSV fragments (`"foo, bar"`). Treating commas as extra separators avoids a tokenizer seeing one giant field.

`nltk.TweetTokenizer` keeps:

- `@`-style mentions (already rewritten to `<user>` in this dump)
- hashtags as a single token (`#not`, `#sarcastictweet`)
- emoji as their own tokens
- punctuation as tokens

Labels are `pandas.read_csv(..., header=None)` then `.squeeze()`, so a one-column file becomes a 1-d array of 0/1.

Return value: `(list[list[str]], np.ndarray, n_lines)`.

The examples package mirrors this with `examples.tokenize.tweet_tokenize` and `examples.tokenize.read_sentence_file` so the same rules can run without NLTK.

## Stage 2a — classical features (`ml_read_data`)

```
ReadOpen
  → AverageVectorPerTweet(tokens, glove)      # 200-d
  → AverageVectorPerEmoji(tokens, emoji2vec)  # 200-d
  → X_emoji = concat(tweet_avg, emoji_avg)    # 400-d
  → shared permutation applied to both views
```

`AverageVectorPerTweet` and `AverageVectorPerEmoji` are the same loop over different vocabs:

- Skip tokens that are not in `model.vocab`.
- If at least one hit: mean of those vectors.
- If zero hits: `np.zeros(200)`.

Gensim 4 renamed `KeyedVectors.vocab` to a key set. The 2023 notebooks assume Gensim 3-style `.vocab` and `__getitem__`. If you upgrade Gensim, use `word in kv` and `kv[word]`.

**Shuffle.** `ml_read_data` draws `np.random.permutation(len(X))` and reorders both views with the same index. Callers who later compare `y` against a Keras sequence model trained on *unshuffled* `ReadOpen` output must not reuse these `y` arrays blindly. The metrics notebook keeps a separate `ReadOpen` path for the deep model for that reason.

## Stage 2b — sequence features (`Preprocess`, `preprocess_test`)

Training:

1. `Tokenizer().fit_on_texts(docs)` on the already-tokenized lists from `ReadOpen`. Keras joins each list back into a string internally, then re-tokenizes on whitespace / punctuation. This is slightly lossy compared to staying on `TweetTokenizer` output.
2. `texts_to_sequences` + `pad_sequences(..., padding='post')`. Train padding length becomes `maxlen` (78 on the saved models).
3. Allocate `embedding_matrix = zeros((count, 200))` where `count` is **the number of training lines**, not `len(tokenizer.word_index) + 1`. That is a 2023 quirk: the matrix is oversized relative to the true vocabulary. It is safe as long as every index the tokenizer emits is `< count`, which it is because the vocab is built from those same lines.
4. Fill each `word_index` row from GloVe, else from emoji2vec (see [emoji-fusion.md](emoji-fusion.md)), else zeros.

Test / subtest:

```python
preprocess_test(tokenizer, maxlen, test_docs)
```

reuses the training tokenizer and the training `maxlen`. Unknown tokens disappear (Keras default). Longer tweets are truncated.

## What the saved Keras graphs expect

`evaluate_loaded_dl_models.ipynb` and the metrics notebook both build sequences this way, then call `model.evaluate(X_test, y_test)`. The saved `sequential_*` graphs have an input length of 78 and an embedding width of 200. If you change padding, they will not load cleanly against new arrays.

Saved graphs live at:

- `model/best_model_single_modal/`  (test acc 0.8635, subtest 0.8669)
- `model/best_model_multi_modal/`   (test acc 0.8735, subtest 0.8921)

Only `keras_metadata.pb` and `saved_model.pb` are in git. Variable shards may be missing depending on how the upload was done; treat the graphs as reference artifacts, not as a guaranteed `tf.keras.models.load_model` success on TensorFlow 2.15+.

## Feature dimensionality cheat sheet

| View | Shape per tweet | Builder |
| --- | --- | --- |
| Token list | variable | `ReadOpen` |
| GloVe average | `(200,)` | `AverageVectorPerTweet` |
| emoji2vec average | `(200,)` | `AverageVectorPerEmoji` |
| Multimodal baseline | `(400,)` | concatenate |
| Padded ids | `(maxlen,)` | `Preprocess` / `preprocess_test` |
| Embedding table | `(count, 200)` | `Preprocess` |
| Bi-LSTM hidden (per step) | `(512,)` | two 256-unit directions |
| Attention context | `(512,)` | `Attention` |
| Logit | `(1,)` | `Dense(sigmoid)` |

## Worked micro-example

Tweet:

```
I just love getting shots 💉 #sarcastictweet
```

After `ReadOpen`-style tokenize:

```
["i", "just", "love", "getting", "shots", "💉", "#sarcastictweet"]
```

| Token | GloVe? | emoji2vec? |
| --- | --- | --- |
| i, just, love, getting, shots | usually yes | no |
| 💉 | no (or rarely) | yes |
| #sarcastictweet | sometimes, as a Twitter-GloVe hashtag | no |

Single-modal average: mean of the five word vectors. Multi-modal average: that 200-d vector plus the 200-d `💉` vector. Sequence model: seven ids, `💉` row of \(E\) is the emoji2vec vector if `get_emoji2vec=True`.

`examples/run_pipeline.py` prints this walk-through on a 16-tweet toy corpus using hash embeddings.

## Dataset hygiene notes

- Some training lines that contain `#Not` are labeled `0`. Case-folding turns `#Not` into `#not`, which is also the dominant sarcasm marker. A purely lexical hashtag rule would mislabel those rows. The models have to use the rest of the tweet.
- Mentions are already `<user>`. Do not expect `@` tokens.
- Quoting is inconsistent. `ReadOpen` strips line endings but does not unquote. Leftover `"` characters become tokens.
- The sentence files are UTF-8 with occasional broken bytes. `ReadOpen` uses `errors="replace"`.
- `inspect_dataset.py` reports length histograms, hashtag rates, and emoji rates so you can see the subtest shift without opening a notebook.
