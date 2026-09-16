# Dataset notes

This course project classifies English tweets as **sarcastic** (`1`) or **not sarcastic** (`0`). Each split is stored as a pair of line-aligned files:

| File | Role |
| --- | --- |
| `dataset/{split}_sentence.csv` | One tweet per line. Lines are not a true multi-column CSV. |
| `dataset/{split}_label.csv` | One integer label per line, same order as the sentence file. |

Splits:

| Split | Tweets | Non-sarcastic (`0`) | Sarcastic (`1`) | Notes |
| --- | ---: | ---: | ---: | --- |
| `train` | 39,780 | 21,292 (53.5%) | 18,488 (46.5%) | Slightly imbalanced; many `@`-style mentions rewritten as `<user>`. |
| `test` | 2,000 | 1,000 (50%) | 1,000 (50%) | Balanced hold-out used for the main numbers in the notebooks. |
| `subtest` | 278 | 106 (38.1%) | 172 (61.9%) | Almost every tweet contains at least one emoji. This is the emoji-co-occurrence slice. |

Counts above were recomputed from the files in this repository. Token-length statistics after the same comma-to-space collapse `ReadOpen` uses, then whitespace split:

| Split | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| `train` | 1 | 16 | 16.46 | 51 |
| `test` | 1 | 16 | 16.31 | 36 |
| `subtest` | 5 | 17 | 17.59 | 36 |

## Why there is a subtest

The research question is not only “does a Bi-LSTM beat an SVM?” It is whether **emoji vectors** help when sarcasm is carried by the *combination* of wording and emoji.

The full test set mixes tweets with and without emoji. The 278-row subtest is the place where that combination is almost always present (276 / 278 tweets contain an emoji code point). If emoji2vec is doing useful work, the gain should show up more clearly here than on the mixed test set. That is exactly what the 2023 numbers show for Random Forest and Bi-LSTM + Attention. See [evaluation.md](evaluation.md).

## Distant-supervision cues in the text

A large fraction of sarcastic tweets in this family of datasets were originally collected because they contained an explicit cue hashtag. Those hashtags are still visible in the files:

| Hashtag | Train count | Test count | Subtest count |
| --- | ---: | ---: | ---: |
| `#not` | 3,191 | 465 | 110 |
| `#yeahright` | 241 | 19 | 5 |
| `#sarcastictweet` | 229 | 38 | 18 |
| `#sarcasm` | (rare in train top-15) | 89 | 0 |

Other frequent tags (`#mtvstars`, `#cantwait`, `#fail`, `#fml`) are topical or affective, not sarcasm labels.

This has two practical consequences:

1. A **lexical cue baseline** that only looks for `#not` / `#sarcasm` / `#yeahright` will look stronger than a fair “language understanding” model, especially on `test` and `subtest`. The example in `examples/cue_baseline.py` exists so that gap is visible.
2. Some `#not` tweets in `train` are labeled `0`. Distant supervision is noisy. Do not treat hashtag presence as ground truth.

## Surface statistics that affect modeling

Recomputed from the current files with the regexes in `examples/common.py` (Unicode emoji ranges plus dingbats and flags; hashtags matching `#\w+`; mentions matching `<user>` or `@\w+`):

| Split | With emoji | With hashtag | With mention | With a sarcasm cue tag |
| --- | ---: | ---: | ---: | ---: |
| `train` | 5,470 (13.8%) | 8,486 (21.3%) | 9,433 (23.7%) | 3,716 (9.3%) |
| `test` | 277 (13.9%) | 932 (46.6%) | 5 (0.2%) | 613 (30.6%) |
| `subtest` | 277 (99.6%) | 155 (55.8%) | 0 (0.0%) | 134 (48.2%) |

Two distribution shifts matter:

- **Mentions.** Train still has thousands of `<user>` tokens. Test and subtest almost never do. A model that overfits mention patterns will not transfer.
- **Hashtags.** Test/subtest are much richer in sarcasm hashtags than train. That inflates any cue-aware method on the official splits.

Emoji rate by label:

| Split | Emoji rate, label `0` | Emoji rate, label `1` | Cue-tag count `0` / `1` |
| --- | ---: | ---: | ---: |
| `train` | 15.0% (3,193) | 12.3% (2,277) | 91 / 3,625 |
| `test` | 10.6% (106) | 17.1% (171) | 0 / 613 |
| `subtest` | 100.0% (106) | 99.4% (171) | 0 / 134 |

On the mixed test set, sarcastic tweets are *more* likely to contain emoji. On train the opposite is true. That is another reason to report subtest separately instead of averaging everything into one accuracy.

On `test` and `subtest`, every sarcasm cue hashtag sits on a positive label. Train is leakier: 91 cue-tagged tweets are labeled `0`. A hashtag-only baseline is therefore an upper-bound cheat on the official hold-outs, not a fair language model.

## What a line looks like

The sentence files keep the original tweet string, including emoji, hashtags, and the `<user>` anonymization. A few examples from `train` / `test` (labels shown in brackets):

```text
[0] <user> i hope youre lurking rn. i want to listen to hallucination & wanna love you again live someday, pretty please?! 😭 😭 😭
[0] 05 really taught me a valuable lesson I'm never gonna be late again! #Not
[1] I loovee when people text back ... 😒 #sarcastictweet
[1] Don't you love it when your parents are Pissed because you were gonna study after bubble soccer ! #IKnowIDo #not 😃 🔫
```

The first sarcastic test tweet is the classic pattern this project is about: positive wording (“I loovee when people text back”), a mismatch emoji (`😒`), and an explicit sarcasm tag.

## File-format quirks

`data_utils.ReadOpen` does **not** parse CSV columns. It:

1. Reads raw lines.
2. Replaces commas with spaces (`' '.join(line.strip().split(','))`).
3. Tokenizes with NLTK `TweetTokenizer`.
4. Lowercases every token.
5. Reads labels with `pandas.read_csv(..., header=None)` and squeezes them to a 1-D array.

Quoted tweets such as `"Late nights, early mornings is how I live my life."` therefore lose the comma and the wrapping quotes after step 2. That is intentional for this repo’s 2023 pipeline, not a pandas dialect.

If you write a new loader, keep line alignment with the label file. Do not use a CSV reader that drops or merges quoted newlines; these files are one record per physical line.

## What is *not* in the repo

- Raw tweet IDs, author handles, or a hydration script.
- The GloVe Twitter 27B 200-d vectors used to build the word embedding matrix. Only `emoji2vec.bin` and `emoji2vec_twitter.bin` are checked in.
- A data card from the original collection paper. Treat this folder as the course snapshot, not as a newly released corpus.

Reproduce the tables in this page with:

```bash
python examples/explore_dataset.py
```
