# Preprocessing

Two feature pipelines share a tokenizer and then diverge.

```text
tweet line
    │
    ├─ ReadOpen: join on commas, TweetTokenizer, lowercase
    │
    ├─ baselines (ml_read_data)
    │     ├─ mean GloVe rows that hit            → 200-d  (single-modal)
    │     └─ mean GloVe ⊕ mean emoji2vec         → 400-d  (multi-modal)
    │
    └─ deep model (Preprocess)
          ├─ Keras Tokenizer → integer ids
          ├─ pad_sequences(..., padding="post")
          └─ embedding matrix, 200-d, frozen
                ├─ GloVe if the token is in GloVe
                ├─ else mean emoji2vec of glyphs in the token
                │     (skipped when get_emoji2vec=False)
                └─ else zeros
```

## Shared tokenizer

[`data_utils.ReadOpen`](../data_utils.py) is the only place tweets become
token lists. The example replica is [`examples/dataset_io.py`](../examples/dataset_io.py)
plus [`examples/tokenize.py`](../examples/tokenize.py).

```python
from examples.dataset_io import read_open_replica

docs, labels, n = read_open_replica(
    "dataset/train_sentence.csv",
    "dataset/train_label.csv",
)
```

`n` is `len(lines)`, which the 2023 deep-model path later passes to
`Preprocess` as `count`. That value is a *document* count, not
`len(tokenizer.word_index) + 1`. The embedding matrix is therefore
allocated `(n_docs, 200)` rather than `(vocab_size, 200)`. It happens to
be large enough on this corpus; do not copy the pattern into new code.

## Baseline features

[`AverageVectorPerTweet`](../data_utils.py) and `AverageVectorPerEmoji`
are the same loop:

1. For each token, if it is in the keyed-vectors table, keep the row.
2. If the tweet had at least one hit, return the mean.
3. Otherwise return `zeros(200)`.

`ml_read_data` then:

* stacks the word means as `X`
* concatenates word means with emoji means as `X_emoji`
* draws one permutation and applies it to both views so the rows stay
  aligned

Gensim 3 membership was `token in model.vocab`. Gensim 4 removed
`.vocab`; use `token in model` (or `token in model.key_to_index`). The
example tables are plain `dict`s so the issue does not arise there.

Walk through a tweet with 8-d stand-in vectors:

```bash
python examples/embedding_demo.py
```

## Deep-model features

[`Preprocess`](../data_utils.py) fits a Keras `Tokenizer` on the training
token lists, converts texts to sequences, and right-pads to the longest
training tweet (`padding="post"`). The evaluation notebooks then call
`preprocess_test(tokenizer, maxlen, test_docs)` so test / subtest share
the training vocabulary and length (78 steps in the saved-model
summaries).

Embedding rows:

| Token situation | Multi-modal (`get_emoji2vec=True`) | Single-modal (`False`) |
| --- | --- | --- |
| in GloVe | GloVe row | GloVe row |
| not in GloVe, emoji glyphs in the token | mean emoji2vec of those glyphs | zeros |
| otherwise | zeros | zeros |

The single- vs multi-modal *deep* models therefore share architecture
and only differ in whether unknown tokens may pick up an emoji vector.
The baseline models differ in input width (200 vs 400) instead.

`build_embedding_matrix` in [`examples/embeddings.py`](../examples/embeddings.py)
is the same decision table on toy vectors.

## Shuffling

`ml_read_data` shuffles with `np.random.permutation` and does **not**
seed the RNG. Re-running the baseline notebooks without the saved
`.pkl` files will not bit-match the 2023 numbers. The deep-model path
does not shuffle inside `Preprocess`; any shuffle would have happened
in a training notebook that was not uploaded.

## External tables

| Table | Dim | Used by | In this repo? |
| --- | ---: | --- | --- |
| GloVe Twitter 27B | 200 | both pipelines | no |
| `emoji2vec_twitter.bin` | 200 | both pipelines | yes |
| `emoji2vec.bin` | 300 originally, project uses a Twitter-aligned 200-d file | unused by the notebooks | yes |

The notebooks load GloVe from the working directory as
`glove.twitter.27B.200d.bin` (binary) or `glove_tt.txt` (text). See
[reproduction.md](reproduction.md).
