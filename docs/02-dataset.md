# 02 — Dataset

Three line-oriented files live in `dataset/`. They are **not** headered
CSVs. Each row of `*_sentence.csv` is one tweet; the matching row of
`*_label.csv` is `0` (literal / non-sarcastic) or `1` (sarcastic).

| Split | Rows | Literal | Sarcastic | Mean tokens | Emoji % | Hashtag % | Cue-hashtag % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 39,780 | 21,292 | 18,488 | 16.99 | 13.78 | 21.33 | 9.38 |
| test | 2,000 | 1,000 | 1,000 | 15.89 | 13.95 | 46.60 | 30.70 |
| subtest | 278 | 106 | 172 | 17.27 | **100** | 55.76 | 48.20 |

Token counts in this table use `examples.common.tokenize`, which approximates
`nltk.TweetTokenizer` (the tokenizer in `data_utils.ReadOpen`). Re-run
`python3 examples/01_explore_dataset.py` to regenerate them.

## How the files were written

`ReadOpen` does two surprising things:

1. `sentence = ' '.join(line.strip().split(','))` — every comma becomes a
   space *before* tokenization. Quoted CSV fields that contain commas are
   therefore split. That is why a tokenizer-level reimplementation has to
   do the same substitution to stay comparable.
2. Every token is lowercased. Hashtags survive as `#not`, `#sarcastictweet`.
   User mentions in this export are already the placeholder `<user>`.

Labels are a single column with no header, loaded in the notebooks with
`pandas.read_csv(..., header=None)` and `.squeeze()`.

## Why there is a subtest

Emoji rate on train/test is only ~14%. A multimodal model that concatenates
a 200-d emoji average will, for the other 86% of rows, concatenate a zero
vector. You cannot see the emoji channel on the full test set; the metric
is dominated by the word channel.

Subtest is the subset of evaluation tweets in which **every** row contains
at least one pictograph (278 rows, 172 sarcastic / 106 literal). That is
also why subtest is *not* balanced: sarcastic tweets in this export are
slightly more likely to carry a face emoji, but the stronger bias is
simply “this slice was filtered on emoji presence”.

The notebooks evaluate every model on both test and subtest. The
interesting comparison is always **W vs WE on subtest**, not W vs WE on
test.

## Cue hashtags and distant supervision

A large fraction of public Twitter sarcasm data is collected by querying
`#sarcasm` / `#sarcastic` / `#not`. Those tags are still in the text.

`examples/03_lexical_cues.py` measures a rule that predicts sarcastic iff a
cue hashtag from `{#sarcasm, #sarcastic, #sarcastictweet, #not, ...}` is
present. On **test**, 61% of sarcastic rows still have a cue tag and
**zero** literal rows do; the rule scores 80.7% accuracy. On **train** cue
tags are rarer (19.7% of sarcastic rows, 0.44% of literal). Hashtag rate
jumps from 21% (train) to 47% (test). Treat test numbers as
“performance on a cue-heavy evaluation slice”, not as in-the-wild
Twitter. The full write-up is [08-control-experiments.md](08-control-experiments.md).

Literal tweets also use `#not` in the ordinary English sense (“100 days
until Christmas! 🌲 #too soon #not ready yet”, label 0 in train). A
hashtag-only rule therefore has non-zero false positives.

## Length and format

Almost all tweets are 5–30 tokens after tokenization. The neural model
pads to the train maximum (`pad_sequences(..., padding='post')`), which the
2023 run logged as length 78 in `model.summary()` — that is the padded
width of the **training** matrix, not a tweet that is 78 words long. A
handful of train rows exceed 40 tokens (one reaches 51).

`<user>` placeholders appear frequently. URLs are already stripped or
never present in this export.

## Class examples (short)

Literal, with emoji:

> My long luscious hair is gone. 😢 😢 😢 😢

Sarcastic, cue hashtag + emoji (subtest):

> I loovee when people text back ... 😒 #sarcastictweet

Sarcastic, no hashtag (harder, and some of these look noisy):

> Imagine how awesome life would be if you could be allergic to homework.

See [07-annotated-examples.md](07-annotated-examples.md) for a longer
hand-annotated sample, including rows that look mislabeled.

## How to load the splits in new code

Prefer `examples.common.io.load_split("train")` for documentation work. It
does not depend on pandas. The original path is:

```python
from data_utils import ReadOpen
docs, labels, n = ReadOpen("dataset/train_sentence.csv", "dataset/train_label.csv")
```

`docs` is a list of token lists, not raw strings.
