# Examples

Self-contained walk-throughs of the math in `data_utils.py` and
`attention_layer.py`. They do **not** load GloVe, emoji2vec, TensorFlow, NLTK,
or sklearn. `numpy` plus the standard library is enough.

The tweets in `toy_corpus.py` are original examples written for this folder.
They are styled like the course subtest (`#not`, weary-face emoji, fake-positive
verbs) but they are not rows copied from `dataset/`.

## What maps onto what

| Example | Original project piece |
| --- | --- |
| `tokenize.tweet_tokenize` | `data_utils.ReadOpen` comma-join + `TweetTokenizer` + lowercase |
| `average_vectors.average_channel` | `AverageVectorPerTweet` / `AverageVectorPerEmoji` |
| `average_vectors.multimodal_features` | `ml_read_data` concatenate → 2× width |
| `hash_embeddings.HashEmbeddings` | `gensim.KeyedVectors` lookup (hashed stand-in) |
| `attention_numpy.attention_forward` | `attention_layer.Attention.call` |
| `toy_classifier.fit_logreg` | *not* a 2023 baseline — a readable stand-in |
| `inspect_dataset.py` | the six files under `dataset/` |

## Run

From the repository root:

```bash
python3 examples/inspect_dataset.py
python3 examples/run_pipeline.py
python3 examples/run_attention.py
python3 examples/run_toy_classifier.py
python3 -m examples.run_all
python3 -m unittest discover -s tests -v
```

## Scripts

### `inspect_dataset.py`

Reads the real CSVs. Prints, for train / test / subtest:

- class balance
- fraction of tweets with an emoji
- fraction with any hashtag
- fraction with `#not` / `#sarcastictweet` / `#sarcasm` / `#yeahright`
- token-length mean / p50 / p90
- top hashtags

This is the fastest way to see why the subtest is the slice where multimodal
fusion moves the 2023 numbers.

### `run_pipeline.py`

Tokenizes `i just love getting shots 💉 #sarcastictweet`, mean-pools a 32-d
text channel and a 32-d emoji channel, concatenates them, then compares

```
i love mondays 😒 #not     (sarcastic)
i love mondays             (literal)
```

The text-channel cosine of that pair is high (same words). The multimodal
cosine drops because the second half is a real emoji vector vs zeros. That is
Fusion A from [docs/emoji-fusion.md](../docs/emoji-fusion.md).

### `run_attention.py`

Builds an 8-step hidden sequence with a spike at step 5, runs the Raffel
attention formula, and prints a bar of `α_t`. Then masks step 5 as pad to show
the mass moving, then a flat sequence that yields uniform alphas.

### `run_toy_classifier.py`

Fits a numpy logistic regressor on the 16-tweet corpus twice: 32-d text only,
then 64-d concatenated. Prints accuracy / precision / recall / F1 and
`P(sarcastic)` on the illustrative pair. Expect the multimodal model to
separate `i love mondays` from `i love mondays 😒 #not` more cleanly.

## Design constraints

- **Deterministic.** Hash embeddings and the logistic seed are fixed. Re-runs
  match.
- **No network.** `inspect_dataset.py` only opens local CSVs.
- **No company code.** Everything here is personal coursework documentation.
- **No silent rewrite of `data_utils.py`.** The 2023 helpers stay as they were;
  these modules are a documented twin so the idea can be tested.

## Adding a tweet to the toy set

Edit `_PAIRS` in `toy_corpus.py`. Keep the illustrative pair intact — the unit
tests look it up by exact string. Re-run `run_pipeline.py` and
`tests/test_toy_classifier.py`.
