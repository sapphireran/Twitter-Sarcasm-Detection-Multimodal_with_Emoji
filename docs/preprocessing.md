# Preprocessing

All original training paths go through `data_utils.py`. There are two representations: a **mean-pooled vector** for sklearn baselines, and a **padded index sequence + embedding matrix** for the Keras model.

## Shared first step: `ReadOpen`

```text
raw line
  → join on commas (commas become spaces)
  → NLTK TweetTokenizer
  → lowercase each token
```

Labels are `pandas.read_csv(..., header=None)` squeezed to a 1-D array. The function also returns `len(lines)` so the Keras embedding table can be allocated as `(line_count, 200)`.

Two consequences of that allocation:

1. Index `0` is padding (Keras `Tokenizer` is 1-based). The extra row is unused.
2. If vocabulary size ever exceeded the line count, later word indices would be out of range. With ~40k tweets that did not happen in the 2023 run.

## Mean-pooled baselines: `ml_read_data`

For each tweet:

1. **Word channel.** Average every token that exists in the GloVe KeyedVectors vocab. Missing tweets become a 200-D zero vector.
2. **Emoji channel.** Average every token that exists in the emoji2vec vocab (typically the emoji characters themselves). Empty emoji lists also become a 200-D zero.
3. **Multimodal vector.** Concatenate the two averages → 400-D.

Both copies of the labels are shuffled with the **same** permutation so word-only and word+emoji rows stay aligned.

Gensim 3.x used `model.vocab` and `model[word]`. Gensim 4.x renamed that to `model.key_to_index` / `model.get_vector`. The committed scripts are written for the 3.x API that the notebooks used in 2023.

## Sequence model: `Preprocess` / `preprocess_test`

`Preprocess` fits a Keras `Tokenizer` on the token lists, pads **to the longest training tweet** (`padding='post'`), and builds a `(count, 200)` embedding matrix:

| Token in GloVe? | Token looks like emoji? | Matrix row |
| --- | --- | --- |
| yes | ignored | GloVe vector |
| no | emoji2vec hit, `get_emoji2vec=True` | mean of those emoji vectors |
| no | emoji2vec hit, `get_emoji2vec=False` | zeros (single-modal ablation) |
| no | no emoji, or emoji2vec KeyError | zeros |

`preprocess_test` reuses the **training** tokenizer and the **training** `maxlen` so evaluation sequences line up with the frozen embedding layer (`input_length=l` in `PrepModel`).

In the recorded evaluation run, `maxlen` was **78**. That is why the saved Sequential models show `(None, 78, 200)` on the first wrapper.

## Single-modal vs multimodal

The architecture in `dl_model.py` does not change. The ablation is entirely in the embedding table:

- **Single-modal (`_w`).** `Preprocess(..., get_emoji2vec=False)` (or an equivalent path that never copies emoji2vec rows).
- **Multimodal (`_we`).** Unknown word pieces that are emoji get an emoji2vec average instead of zeros.

Classical baselines concatenate mean vectors instead of mixing them inside the same 200-D table. That is a different fusion rule: early concatenation (400-D) versus in-vocabulary substitution (still 200-D). The notebooks still call both settings “single-modal” / “multi-modal”.

## What the example scripts do instead

`examples/sarcasm_lab` does **not** load GloVe or emoji2vec. It:

- reads the same line-aligned files
- applies a TweetTokenizer-like regex splitter (no NLTK)
- extracts hashtags, emoji, and a small sarcasm-cue lexicon

That is enough to inspect the data and train a lexical baseline. It is **not** a drop-in replacement for `ml_read_data`.
