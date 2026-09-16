# Preprocessing

All official training and evaluation paths go through `data_utils.py`. There are two pipelines that share a tokenizer and then diverge.

```text
raw line
  -> strip
  -> replace commas with spaces
  -> NLTK TweetTokenizer
  -> lowercase
        |-- classical ML --> average GloVe / emoji2vec rows
        |-- deep model   --> Keras Tokenizer + pad + embedding matrix
```

## Shared tweet tokenization

`ReadOpen(filename, Labelfile)` is the only place raw text is turned into token lists.

```python
sentence = ' '.join(line.strip().split(','))
for token in tokenizer_tweet.tokenize(sentence):
    temp.append(token.lower())
```

Why this sequence exists:

| Step | Effect on sarcasm tweets |
| --- | --- |
| Strip | Drops the trailing newline. Windows `\r` is not specially handled. |
| Split on commas | Turns `"hello, world"` into `hello  world`. Prevents Keras/`pandas` CSV quoting from becoming a second parser. |
| `TweetTokenizer` | Keeps emoji, hashtags, and emoticons as tokens instead of splitting `😒` or `#not` into junk. |
| Lowercase | Makes `#Not` and `#not` the same token. GloVe Twitter is mostly lowercased. |

`TweetTokenizer` is the important part. A naive `str.split()` will smash emoji clusters and treat `#sarcastictweet` as a normal word only by accident. The example `examples/tokenize_tweets.py` prints both behaviors on the same lines.

User mentions in this snapshot are already rewritten as the literal token `<user>`. The tokenizer will keep that as one token.

## Classical ML path (`ml_read_data`)

Used by `baseline_models.ipynb` and the sklearn half of `get_metrics_of_models.ipynb`.

1. Tokenize with `ReadOpen`.
2. `AverageVectorPerTweet`: mean of every token that exists in the GloVe KeyedVectors. Missing words are skipped. If *no* token hits, the tweet becomes a 200-d zero vector.
3. `AverageVectorPerEmoji`: same loop against the emoji2vec table. Most word tokens miss. Emoji tokens hit. Empty tweets become another 200-d zero vector.
4. Single-modal features `X` = tweet-average GloVe only (200-d).
5. Multi-modal features `X_emoji` = `concat(glove_avg, emoji_avg)` (400-d).
6. Labels are copied, then **both** views are shuffled with the same `numpy.random.permutation`. There is no seed in the original function.

Implications:

- Frequency information inside a tweet is washed out. Repeated `😭 😭 😭` still contributes three identical rows to the mean, so intensity is not fully discarded.
- A tweet with no in-vocab words and no emoji is a zero vector. SVM / trees can still use that as “unknown”.
- Because shuffle is unseeded, two runs of `ml_read_data` do not produce the same row order. The saved `.pkl` baselines were trained on one such shuffle; reloading them is safer than refitting if you want the notebook numbers.
- Concatenation is *early fusion*. The classifier sees 400 independent dimensions. There is no cross-attention between words and emoji in the sklearn models.

## Deep-learning path (`Preprocess` / `preprocess_test`)

Used by `evaluate_loaded_dl_models.ipynb` and the Bi-LSTM half of `get_metrics_of_models.ipynb`.

1. Fit a Keras `Tokenizer` on the *training* token lists. Default Keras filters will strip punctuation; emoji and `#hashtags` that survived `TweetTokenizer` generally remain.
2. Convert texts to integer sequences and `pad_sequences(..., padding='post')`. Train length `l` becomes the model’s `input_length`.
3. Build a `(vocab_size, 200)` embedding matrix:
   - If the token is in GloVe, copy that 200-d row.
   - Else extract emoji code points from the token with the `emoji` package.
   - If `get_emoji2vec=True` (the multi-modal setting), average those emoji2vec rows into the same 200-d slot.
   - Otherwise write zeros (the single-modal setting for OOV / emoji-only tokens).
4. `preprocess_test` reuses the **train** tokenizer and the **train** pad length. Test tokens unseen at train time become `0` (padding / OOV).

`PrepModel` then freezes that matrix (`trainable=False`). The network never updates GloVe or emoji2vec; it only learns LSTM and attention weights on top.

### Single-modal vs multi-modal in the same 200-d space

This is easy to miss. Classical ML concatenates to 400-d. The Bi-LSTM does **not**. Both variants use one 200-d embedding channel:

| Setting | GloVe hit | Emoji / OOV token |
| --- | --- | --- |
| Single-modal (`get_emoji2vec=False`) | 200-d GloVe | zeros |
| Multi-modal (`get_emoji2vec=True`) | 200-d GloVe | mean emoji2vec, also 200-d |

Emoji and words therefore live in one sequence. The Bi-LSTMs can, in principle, learn that `love` followed by `😒` is a different pattern than `love` followed by `😍`. Averaged sklearn features cannot represent that order.

### Sequence length and the attention bias

`Attention.build` allocates a bias of shape `(input_shape[1],)` — one scalar per time step. That ties the layer to the padded length used at train time. You cannot change `maxlen` at inference without rebuilding the layer. The saved models in `model/best_model_*` expect the train pad length from the 2023 run (78 steps in the notebook summary).

## Test-time contract

To score new tweets the way the notebooks do:

1. Run them through the same comma-strip + `TweetTokenizer` + lowercase steps.
2. Encode with the **train** tokenizer, not a tokenizer refit on the new texts.
3. Pad / truncate to the **train** `maxlen`.
4. For sklearn models, average against the same GloVe / emoji2vec tables and concatenate if the pickle is a `*_we` / multi-modal model.

`examples/tokenize_tweets.py` stops at step 1 on purpose: it does not require GloVe, TensorFlow, or NLTK. Use it to inspect how a tweet will look *before* the embedding tables are applied.

## Things this pipeline does not do

- No URL / mention / hashtag stripping beyond what the tokenizer already emits.
- No spelling correction for elongated affect (`loovee`, `THANK YOOOOOOO`). Those forms are useful sarcasm cues; leaving them is correct.
- No class weighting despite the mild train imbalance.
- No seeded shuffle and no explicit validation split in `data_utils.py`. The notebooks treat `test` as `X_val` when they need a validation handle.

If you extend this project, keep those omissions in mind before comparing a new number to the 2023 table.
