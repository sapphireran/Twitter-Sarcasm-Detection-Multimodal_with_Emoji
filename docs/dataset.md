# Dataset

The shipped files under `dataset/` are aligned sentence/label pairs. Each
`.csv` is a bare column: one tweet per line in `*_sentence.csv`, one `0`/`1`
in the matching `*_label.csv`. There is no header row.

## Split sizes

Counts below were recomputed from the files in this checkout (September
2026). They match the notebook evaluation sizes (2,000 test rows, 278
subtest rows).

| File pair | Rows | Sarcastic (`1`) | Non-sarcastic (`0`) | ≥1 emoji | ≥1 hashtag |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_sentence.csv` / `train_label.csv` | 39,780 | 18,488 (46.5%) | 21,292 (53.5%) | 5,470 (13.8%) | 8,486 (21.3%) |
| `test_sentence.csv` / `test_label.csv` | 2,000 | 1,000 (50.0%) | 1,000 (50.0%) | 277 (13.9%) | 932 (46.6%) |
| `subtest_sentence.csv` / `subtest_label.csv` | 278 | 172 (61.9%) | 106 (38.1%) | 277 (99.6%) | 155 (55.8%) |

The subtest is not a random slice of the test set. It is almost the emoji-
containing subset: 277 of 278 rows include an emoji codepoint under the
regex in `examples/lib/tokenize.py`. Class balance also shifts (62%
sarcastic vs 50% on the main test set), so accuracy on the subtest is not
directly comparable to test accuracy.

`examples/01_dataset_preview.py` reprints these counts.

## How the original loaders read the files

`data_utils.ReadOpen`:

1. Opens the sentence file as UTF-8 with `errors="replace"`.
2. Replaces commas inside a line with spaces (`' '.join(line.strip().split(','))`).
3. Tokenises with `nltk.TweetTokenizer` and lowercases every token.
4. Reads labels with `pandas.read_csv(..., header=None)` and squeezes to 1-d.

That comma rewrite is a leftover of an earlier export format. Most lines in
the current files are ordinary tweets and do not contain field commas;
quoted tweets that do will have those commas turned into spaces before
tokenisation.

`ml_read_data` then shuffles with `numpy.random.permutation`. There is no
seed, so classical-model training order is not deterministic unless the
caller sets one.

The example library in `examples/lib/io.py` does **not** shuffle and does
**not** rewrite commas. It is for inspection and the TF-IDF demo, not a
drop-in replacement for the 2023 training path.

## Surface cues and leakage

Hashtag counts on the **test** set (casefolded):

| Hashtag | Sarcastic | Non-sarcastic |
| --- | ---: | ---: |
| `#not` | 465 | 0 |
| `#sarcasm` | 89 | 0 |
| `#sarcastictweet` | 38 | 0 |
| `#yeahright` | 19 | 0 |

On **train**, `#not` occurs 3,191 times across 3,188 tweets (3,105
sarcastic / 83 not by tweet). Other high-precision tags include
`#yeahright` (237 / 4) and `#sarcastictweet` (228 / 1).

This is a known property of hashtag-supervised Twitter sarcasm corpora: the
label often *is* the tag. A model that sees `#not` can look strong without
understanding polarity flips. The BiLSTM and the TF-IDF example both have
access to those tokens. When you compare models, treat hashtag recall as a
ceiling, not as evidence of pragmatic reasoning.

`examples/02_lexical_cues.py` prints precision/recall-style coverage for a
small tag list and for emoji presence.

## Emoji, by class

Emoji are less of a pure leak than `#not`, but they are not neutral either.

**Train set**, most frequent emoji:

| Glyph | In sarcastic tweets | In non-sarcastic tweets |
| --- | ---: | ---: |
| 😂 | 794 | 1,074 |
| 😊 | 379 | 289 |
| 😒 | 295 | 122 |
| 😭 | 241 | 386 |
| 😍 | 187 | 439 |

The unamused face (😒) leans sarcastic; hearts and heart-eyes lean sincere.
The face with tears of joy is common in both classes. That is exactly the
kind of co-occurrence the multi-modal model was meant to use: the same
glyph in different textual neighbourhoods.

On the **test** set the sarcastic side is richer in 😒, 😊, 😅, and 🔫; the
non-sarcastic side is richer in 😂, 😭, and ❤.

## Labelled examples from the test file

These are real rows. Spelling and punctuation are unchanged.

Sarcastic (`1`):

- `I loovee when people text back ... 😒 #sarcastictweet`
- `Don't you love it when your parents are Pissed because you were gonna study after bubble soccer ! #IKnowIDo #not 😃 🔫`
- `feeling like a million bucks after that chem 2 test . 😅 #not`
- `I love walking to school 😄 #SarcasticTweet`

Non-sarcastic (`0`) examples tend to be straightforward reports or
compliments without a polarity tag. The first train rows are mostly class
`0` (concert wishes, thanks, walk recaps). Open the CSVs or run the preview
script if you want a longer sample.

## Embedding files that belong with the data

| File | Header | Role |
| --- | --- | --- |
| `emoji2vec_twitter.bin` | `1661 200` | used with GloVe Twitter 200-d |
| `emoji2vec.bin` | `1661 300` | original Eisner et al. 300-d dump |

Both are word2vec **binary** files. The neural preprocessor only consults
them for tokens that fail the GloVe lookup and then look like emoji.
`examples/03_emoji_vectors.py` reads the 200-d file with a small parser so
you can inspect neighbours without Gensim. Heart-like glyphs in that table
use the emoji-style form `❤️` (U+2764 plus variation selector-16), not
the bare `❤`.

The GloVe Twitter 27B 200-d vectors are **not** in the repo. See
[`reproduction.md`](reproduction.md).

## Recommended checks before training

1. `wc -l dataset/*.csv` — sentence and label files must have the same
   line count. They do in this checkout.
2. Labels must be only `0` and `1`. `tests/test_dataset_integrity.py`
   asserts that.
3. If you re-export the CSVs, keep them headerless. `ReadOpen` and the
   example loader both assume that.
