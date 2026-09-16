# Embeddings

The project treats a tweet as two aligned signals:

1. **Words** — GloVe vectors trained on Twitter.
2. **Emoji** — emoji2vec vectors that live in a word2vec-shaped table.

Classical models concatenate a mean word vector and a mean emoji vector. The Bi-LSTM writes both kinds of vector into **one** 200-d sequence so order is preserved. See [architecture.md](architecture.md) for how those tables are consumed.

## GloVe Twitter 27B, 200-d

The notebooks load:

```text
glove.twitter.27B.200d.bin          # baseline_models.ipynb, evaluate_loaded_dl_models.ipynb
glove_tt.txt                        # get_metrics_of_models.ipynb (text word2vec format)
```

Neither file is in this repository. You need the official [GloVe Twitter crawl](https://nlp.stanford.edu/projects/glove/) (`glove.twitter.27B.200d.txt`) and, if you want the `.bin` path, a one-time conversion with Gensim:

```python
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec("glove.twitter.27B.200d.txt", "glove.twitter.27B.200d.w2v.txt")
kv = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.w2v.txt", binary=False)
kv.save_word2vec_format("glove.twitter.27B.200d.bin", binary=True)
```

Properties that matter for this dataset:

| Property | Why it matters here |
| --- | --- |
| 200 dimensions | Every frozen embedding row in `PrepModel` is 200-d. |
| Twitter crawl | Elongated spellings, `@`-style mentions, and informal casing are closer to this table than to Wikipedia GloVe. |
| Mostly lowercased | Matches `ReadOpen`, which lowercases every token. |
| No emoji coverage to speak of | Face and object emoji almost always miss in GloVe. That miss is the whole reason emoji2vec exists in this project. |

If a token is not in GloVe, `AverageVectorPerTweet` skips it. `Preprocess` then tries to recover emoji code points from the same token and, in the multi-modal setting, writes their mean emoji2vec row instead of zeros.

## emoji2vec

Two binaries are checked in. They share a 1,661-emoji vocabulary but **do not live in the same vector space**:

| File | Dims | Typical use in this repo |
| --- | ---: | --- |
| `emoji2vec_twitter.bin` | 200 | The table the notebooks actually load. Same width as GloVe Twitter 200-d, so it can sit in the Bi-LSTM embedding matrix. |
| `emoji2vec.bin` | 300 | Upstream-style emoji2vec dump (Google-News word2vec width). Useful as a neighbour-list reference, **not** as a drop-in row in `PrepModel`. |

Both are standard little-endian word2vec binaries (one header line `vocab_size vector_size`, then `token` + `vector_size` float32 values). `examples/inspect_emoji2vec.py` reads them **without Gensim** so you can inspect vocab size, norms, and nearest neighbours in a minimal environment.

Do not concatenate or cosine-compare a 200-d Twitter row with a 300-d upstream row. Neighbour lists *inside* each file are the fair comparison. On this snapshot, `😒` sits near `😞 / 🙁 / 😟` in the Twitter table — that is the affect geometry the multi-modal model is buying.

The original method is Eisner, Rocktäschel, Augenstein, Bošnjak, and Riedel, *emoji2vec: Learning Emoji Representations from their Description* (arXiv:1609.08359). Each emoji is trained so that its vector sits near the GloVe vectors of words in its Unicode name / short description (`grinning face`, `unamused face`, `pistol`, …). After that training, `😒` is closer to *unamused* / *annoyed* than to *happy*.

That is the geometric claim this project depends on: **an emoji can fill the same 200-d slot as a word and still carry affect.**

## How the two tables are combined

### Mean-pool + concat (sklearn baselines)

```text
tweet tokens  --hit GloVe---->  mean 200-d   ┐
                                              ├─ concat → 400-d row
tweet tokens  --hit emoji2vec--> mean 200-d  ┘
```

- Words that miss GloVe do not enter the first mean.
- Words that miss emoji2vec do not enter the second mean (this is almost all words).
- A tweet with no in-vocab emoji becomes `concat(glove_mean, 0_200)`.
- A tweet with no in-vocab words becomes `concat(0_200, emoji_mean)`.

This is bag-of-vectors. `love … 😒` and `😒 … love` are the same 400-d point.

### Shared sequence (Bi-LSTM)

```text
token[t] in GloVe?     yes → embedding[t] = GloVe[token]
token[t] has emoji?    yes → embedding[t] = mean(emoji2vec[c] for c in token)
otherwise                   embedding[t] = 0_200
```

`get_emoji2vec=False` (single-modal) writes zeros for the emoji / OOV branch. `get_emoji2vec=True` (multi-modal, the default of `Preprocess`) writes the emoji mean into the **same** 200-d row. The two Bi-LSTM layers then see a time series in which `love` and `😒` are neighbouring steps.

Because `Embedding(..., trainable=False)`, neither table is fine-tuned on this sarcasm data. Any gain from the multi-modal matrix is a gain from *better initial features*, not from jointly training emoji space.

## Practical lookup rules

1. Always lowercase before GloVe lookup (already done in `ReadOpen`).
2. Look up emoji as the character itself (`😒`), not as `:unamused_face:`. emoji2vec keys are the Unicode characters.
3. A token that is *only* an emoji will miss GloVe and hit emoji2vec.
4. A token that mixes text and emoji (rare after `TweetTokenizer`) is handled in `Preprocess` by `emoji.emoji_list(word)` and a second pass that keeps `emoji.is_emoji(c)` characters.
5. Clustered emoji `😭 😭 😭` become three tokens and therefore three rows. Mean-pooling then leans toward that emoji; the LSTM sees a short run of the same vector.

## What the inspector is for

```bash
python examples/inspect_emoji2vec.py
python examples/inspect_emoji2vec.py --compare
python examples/inspect_emoji2vec.py --neighbors 😒 😍 🔫 🌲
```

Use it to confirm, before you install TensorFlow:

- `emoji2vec_twitter.bin` is 200-d (GloVe-Twitter width); `emoji2vec.bin` is 300-d and L2-normalized;
- affect-bearing faces (`😒`, `😍`, `😭`) have stable nearest neighbours in both tables;
- the two files share 1,661 keys but must not be cosine-compared across spaces.

If neighbours look random, the file is truncated or you are decoding UTF-8 keys incorrectly. The custom reader in `examples/common.py` skips the newline-as-separator quirk the same way Gensim does.

Neighbour lists from this snapshot (`emoji2vec_twitter.bin`):

| Query | Nearest keys (cosine) |
| --- | --- |
| 😒 | 😞 0.51, 🙎 0.49, 🙁 0.48, 😟 0.48 |
| 😍 | 😻 0.64, 💌 0.58, 💗 0.55, 😊 0.51 |
| 😭 | 😿 0.66, 😢 0.62, 😞 0.52, 😂 0.51 |
| 🌲 | 🎄 0.63, 🌳 0.48, 🌸 0.48 |

That is the geometry `Preprocess` copies into the frozen embedding matrix when `get_emoji2vec=True`.

## Files you still have to fetch

| Needed for | File | In repo? |
| --- | --- | --- |
| Re-running sklearn or Bi-LSTM notebooks | GloVe Twitter 200-d | No |
| Inspecting emoji geometry / multi-modal matrix | `emoji2vec_twitter.bin` | Yes |
| Comparing to upstream emoji2vec | `emoji2vec.bin` | Yes |

The saved Keras models under `model/best_model_*` already contain a frozen embedding matrix from the 2023 run. Loading them for evaluation still goes through `Preprocess` in the notebooks because the notebooks rebuild `X_test` from raw text rather than storing padded sequences.
