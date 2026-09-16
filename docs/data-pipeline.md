# Data pipeline

This note describes the on-disk splits and the two feature paths implemented
in `data_utils.py`. The example scripts in `examples/` reimplement the
parts that can run without GloVe, gensim, or NLTK so the same story can be
inspected on a laptop.

## Splits

| Split | Sentences | Labels `0` (literal) | Labels `1` (sarcastic) | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| `dataset/train_*.csv` | 39,780 | 21,292 | 18,488 | 46.48% |
| `dataset/test_*.csv` | 2,000 | 1,000 | 1,000 | 50.00% |
| `dataset/subtest_*.csv` | 278 | 106 | 172 | 61.87% |

Each split is a pair of files with the same number of lines:

- `*_sentence.csv` — one tweet per line, already somewhat normalized
  (`<user>` placeholders, spaces around punctuation, emoji kept as characters)
- `*_label.csv` — a single integer `0` or `1` per line, no header

The files are **not shuffled on disk**. Test and subtest are a single
sarcastic block followed by a single literal block (1,000 + 1,000 and
172 + 106). Train is *mostly* grouped — a long literal run, then a long
sarcastic run — but it is not a clean two-block file: there are 13 class
changes and a handful of sarcastic rows inside the literal prefix.
`ml_read_data` therefore draws a random permutation before returning
arrays. Any new training script must do the same or a sequential iterator
will see one class for thousands of steps.

## What the subtest is

The 278-tweet subtest is the **emoji-bearing subset of the official test
set**. Every subtest line contains at least one emoji code point, the class
counts match the emoji-containing rows of `test_label.csv` (172 sarcastic,
106 literal), and the texts are the same tweets.

That construction is why multi-modal gains show up more clearly on subtest
than on the full test set: the full test set is 86% emoji-free, so a 200-d
emoji average is a zero vector for most rows.

The walkthrough `examples.inspect_dataset` re-checks this identity on
every run: every subtest row occurs in `test` in the same relative order,
and those rows are exactly the test tweets that contain an emoji
code point (including dingbats such as ⭕️).

Approximate surface-cue rates computed from the checked-in CSVs:

| Cue | Train | Test | Subtest |
| --- | ---: | ---: | ---: |
| At least one `#` hashtag | 8,502 | 935 | 156 |
| `@` / `<user>` mention | 9,500 | 5 | 0 |
| At least one emoji | 5,479 | 278 | 278 |
| `#not` (case-insensitive) | 3,491 | 515 | 116 |
| Token containing `sarcas` | 449 | 140 | 19 |

Train tweets are short: median 16 whitespace tokens, max 51. Test is similar
(median 16, max 36). The Keras path pads to the training maximum width.

## `ReadOpen`

```text
line
  → strip
  → split on commas and re-join with spaces
  → NLTK TweetTokenizer
  → lowercase every token
```

Comma joining is a leftover from an earlier CSV layout. Current sentence
files are already one tweet per line, so the join is usually a no-op unless
a tweet still contains a literal comma.

The return value is `(token_lists, label_array, line_count)`. `line_count`
is reused later as the first dimension of the Keras embedding matrix. That
is a historical shortcut: the matrix should be sized to `vocab_size + 1`,
not to the number of tweets. It works here only because the training
vocabulary is smaller than 39,780. See [architecture.md](architecture.md).

## Classical path: mean pooling

`AverageVectorPerTweet` and `AverageVectorPerEmoji` walk the same token
list. Each function looks the token up in **its own** KeyedVectors table
and averages the hits.

- Missing from that table → skip the token
- No hits at all → write a 200-d zero vector

`ml_read_data` then builds two views:

| Name | Width | Contents |
| --- | ---: | --- |
| `X` | 200 | mean GloVe of in-vocabulary word tokens |
| `X_emoji` | 400 | `X` concatenated with the mean emoji2vec of in-vocabulary emoji tokens |

Both views are shuffled with the **same** index permutation so row `i` stays
aligned across the single-modal and multi-modal experiments.

This is early fusion by concatenation. There is no learned gate between the
word channel and the emoji channel; a linear model or a tree has to discover
that the second 200 dimensions are often zero.

## Neural path: tokenizer + frozen matrix

`Preprocess(docs, count, glove_model, emoji2vec_model, get_emoji2vec=True)`:

1. Fit a Keras `Tokenizer` on the already-tokenized training lists.
2. Convert texts to integer sequences and pad with zeros on the right.
3. Allocate `embedding_matrix` of shape `(count, 200)`.
4. For each `word, i` in `tokenizer.word_index`:
   - if `word` is in GloVe, copy that row
   - else extract emoji characters from `word` with the `emoji` package and
     average their emoji2vec rows when `get_emoji2vec` is true
   - else write zeros

`preprocess_test` reuses the **training** tokenizer and the **training**
pad width so evaluation ids stay in the same space.

The `get_emoji2vec` flag is how the notebooks build the two neural variants
without changing the architecture:

- single-modal (`_w`): unknown tokens, including emoji, stay at zero
- multi-modal (`_we`): emoji tokens that miss GloVe still get an emoji2vec
  row in the same 200-d embedding table

That is a different fusion strategy from the classical 400-d concatenation.
The neural model never sees a second channel; emoji and words share one
sequence of 200-d vectors.

## Why two fusion styles

| | Classical | Neural |
| --- | --- | --- |
| Word signal | mean of word rows | per-token GloVe, order preserved |
| Emoji signal | mean of emoji rows | same table, emoji rows mixed into the sequence |
| Fusion | concatenate 200 + 200 | shared embedding space |
| Empty emoji tweet | 200 zeros appended | sequence is unchanged except missing emoji rows |

The examples package mirrors both ideas with tiny synthetic vectors so you
can print the shapes without downloading GloVe. See
`examples/multimodal_fusion.py` and `examples/embedding_matrix_walkthrough.py`.

## Practical loading rules

Use the helpers in `examples/dataset_io.py` when you only need the CSVs:

```python
from examples.dataset_io import load_split, iter_splits

train = load_split("train")
print(train.n, train.positive_rate, train.sentences[0])
```

Do not assume a header, quoted CSV, or UTF-8 that is clean enough for the
default `open()` encoding. The 2023 reader used `encoding="utf-8"` and
`errors="replace"`. The example loader does the same.
