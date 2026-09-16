# Preprocessing

All feature construction used by the original notebooks lives in `data_utils.py`. This page walks that module in the order a tweet actually travels.

## 1. `ReadOpen(filename, Labelfile)`

```text
raw line
  → join on commas (undo CSV splitting of the tweet text)
  → nltk.TweetTokenizer
  → lowercase every token
  → list[list[str]]
```

Labels are loaded with pandas as a 1-d integer vector. The function also returns `len(lines)`, which later becomes the Keras vocabulary-allocation hint `count`.

`TweetTokenizer` is the important choice. A whitespace split would smash `#not` into the same bucket as `not`, and would break emoji clusters. The NLTK tweet tokenizer keeps hashtags, emoji, and emoticons as tokens, which is what both GloVe-Twitter and emoji2vec expect.

## 2. Averaged embeddings for classical models

`ml_read_data` builds two matrices from the same token lists:

| Matrix | Builder | Width | Meaning |
| --- | --- | ---: | --- |
| `X` | `AverageVectorPerTweet` | 200 | mean of GloVe vectors for in-vocabulary tokens |
| `X_emoji` | tweet mean **concat** emoji mean | 400 | `[gloverep ; emoji2vec_rep]` |

Rules shared by both averagers:

- A token contributes only if it is in `model.vocab` (Gensim KeyedVectors).
- If a tweet has **no** in-vocabulary tokens, the row is a zero vector of length 200.
- The mean is an unweighted arithmetic mean. Frequent function words and rare content words count the same.

After both matrices exist, `ml_read_data` draws one permutation and applies it to `(X, y)` and `(X_emoji, y_emoji)`. That keeps the multimodal pair aligned.

### What this representation can and cannot see

An average is permutation-invariant. “I love waiting in line #not” and “#not I love waiting in line” become the same vector if the token set is the same. Word order, which attention later exploits, is discarded here. That is why the sklearn baselines are a **ceiling on bag-of-embeddings**, not a ceiling on the task.

The 400-d multimodal vector is also a crude fusion: it cannot say *which* word an emoji attached to. It only says “this tweet had these word directions and these emoji directions.”

## 3. Sequence embeddings for the deep model

`Preprocess(docs, count, glove_model, emoji2vec_model, get_emoji2vec=True)` is the Keras path.

1. Fit a `keras_preprocessing.text.Tokenizer` on the tokenized training docs.
2. Convert docs to integer sequences and **post-pad** to the longest training tweet. In the saved multimodal model summary that length is **78**.
3. Allocate `embedding_matrix` with shape `(count, 200)`.
4. For each `word, i` in `tokenizer.word_index`:
   - if `word` is in GloVe, copy that 200-d row;
   - else, run `emoji.emoji_list(word)`, keep characters that `emoji.is_emoji`, look those up in emoji2vec, and write the mean;
   - if `get_emoji2vec` is false, skip the emoji fallback and write zeros;
   - on any lookup failure, write zeros and increment a `nf` (not-found) counter that is currently unused.

`preprocess_test(tokenizer, maxlen, test_docs)` reuses the **training** tokenizer and the **training** pad length. Test tokens that were never seen become zeros in the sequence (Keras default). There is no UNK token with a learned vector.

### Single-modal vs multimodal in this path

The deep single-modal run is the same architecture with `get_emoji2vec=False` (or an embedding matrix that never received emoji2vec rows). The multimodal run writes emoji vectors into the same 200-d table, so an emoji token is not a random OOV — it lands near other emoji in emoji2vec space.

That is a different fusion story from the classical 400-d concatenation. The LSTM sees emoji **in position**, as tokens in the tweet, sharing the same hidden state as words.

## 4. What is *not* done

The 2023 code does not:

- strip sarcasm hashtags
- expand contractions beyond what `TweetTokenizer` already does
- lemmatize or stem
- replace URLs with a single `URL` token (beyond whatever the original dump already did)
- balance the training set
- build character-level or byte-pair features
- fine-tune GloVe (`Embedding(..., trainable=False)` in `PrepModel`)

Those omissions are documented because they affect any follow-up. If you strip `#not` today, expect the heuristic baseline and probably the sklearn models to drop; the LSTM may drop less if it has already learned the surrounding polarity flip.

## 5. Worked toy example

Tweet:

```text
I just love having grungy ass hair 😑 #not
```

Approximate tokens after `ReadOpen`:

```text
i  just  love  having  grungy  ass  hair  😑  #not
```

Classical single-modal row: mean of GloVe(`i`, `just`, `love`, …) ignoring 😑 if it is not in GloVe.

Classical multimodal row: that 200-d mean concatenated with emoji2vec(`😑`) (or zeros if the emoji model misses it). `#not` still lives in the word channel if GloVe-Twitter contains it.

Deep multimodal sequence: integer ids for each token, padded to 78, with the embedding row for 😑 coming from emoji2vec and the row for `#not` coming from GloVe if present.

`examples/tokenize_demo.py` and `examples/embedding_average_demo.py` replay this without requiring Gensim.
