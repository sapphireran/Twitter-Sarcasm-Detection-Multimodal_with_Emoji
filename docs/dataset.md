# Dataset

The files under [`dataset/`](../dataset/) are the only labeled tweets you need
to reproduce the 2023 course numbers. They are already anonymized (`<user>`
in place of mentions) and split the way the notebooks expect.

## Files

| File | Rows | Role |
| --- | ---: | --- |
| `dataset/train_sentence.csv` | 39,780 | Training tweets, one per line |
| `dataset/train_label.csv` | 39,780 | `0` = not sarcastic, `1` = sarcastic |
| `dataset/test_sentence.csv` | 2,000 | Held-out test tweets (balanced) |
| `dataset/test_label.csv` | 2,000 | Test labels |
| `dataset/subtest_sentence.csv` | 278 | Emoji-bearing slice of the test set |
| `dataset/subtest_label.csv` | 278 | Subtest labels |

There is no header row. Sentence files are *almost* one raw tweet per line.
A handful of tweets contain commas and are wrapped in CSV quotes.

## How the 2023 code read the files

[`data_utils.ReadOpen`](../data_utils.py) does **not** use the `csv` module.
It rebuilds each line as:

```python
sentence = " ".join(line.strip().split(","))
```

and then runs `nltk.TweetTokenizer` plus `.lower()` on every token.

That means a quoted tweet such as

```text
"So many useless classes , great to be student"
```

is seen by the model as

```text
"So many useless classes   great to be student"
```

with the quotes still attached to the first and last tokens. The analysis
scripts in [`examples/`](../examples/) expose both behaviors:

* `read_open_replica` — same comma-split as 2023
* `read_sentence_strings` — `csv.reader`, keeps the comma

Run `python examples/tokenize_demo.py` to see the difference on a real line.

## Label balance

Computed by `python examples/dataset_overview.py` from the CSVs in this
repository (not from memory). Token counts use the example tokenizer, so
`<user>` is one token; they will not match an NLTK `TweetTokenizer` dump.

| split | n | non-sarc | sarc | sarc% | emoji | emoji% | mean tok | #not | sarc-tag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 39780 | 21292 | 18488 | 46.5 | 5479 | 13.8 | 18.01 | 3188 | 287 |
| test | 2000 | 1000 | 1000 | 50.0 | 278 | 13.9 | 16.90 | 465 | 130 |
| subtest | 278 | 106 | 172 | 61.9 | 278 | 100.0 | 17.99 | 110 | 19 |

`subtest` is not an independent draw. It is the subset of `test` that
contains emoji (278 of 278 rows match the detector in
`examples/emoji.py`). That is why the notebooks report a second, smaller
score: it is the only slice where an emoji2vec channel can move the
decision. Recompute the table with:

```bash
python examples/dataset_overview.py
```

## Surface cues

Sarcasm in this corpus is often *announced* by a hashtag. Those tags are
still just tokens to GloVe / the BiLSTM; they are not stripped.

Tracked tags: `#not`, `#sarcasm`, `#sarcastic`, `#sarcastictweet`, `#yeahright`.

| split | label | n | emoji | sarc-tag | both | neither |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | 0 | 21292 | 3198 | 91 | 21 | 18024 |
| train | 1 | 18488 | 2281 | 3625 | 701 | 13283 |
| test | 0 | 1000 | 106 | 0 | 0 | 894 |
| test | 1 | 1000 | 172 | 613 | 134 | 349 |
| subtest | 0 | 106 | 106 | 0 | 0 | 0 |
| subtest | 1 | 172 | 172 | 134 | 134 | 0 |

Tag frequency from the same script: train is `#not`×3191, `#yeahright`×241,
`#sarcastictweet`×229, `#sarcastic`×58. Test adds `#sarcasm`×89, which
almost never appears in train under that exact spelling.

Almost every tracked sarcasm hashtag sits on a sarcastic row (91
exceptions in 39,780 training tweets). That is useful context when you
read the high subtest scores: the emoji slice is also the slice where
those tags cluster.

Recompute the table with:

```bash
python examples/sarcasm_cues.py
```

Most frequent pictographs in train: 😂×1868, 😊×668, 😭×627, 😍×626, 😒×417,
❤️×381, 😩×371, 😘×243. Test / subtest share the same head of the list
(😂, 😒, 😊, 😭, 😅) because subtest *is* the emoji-bearing test rows.

## Examples from the files

Sarcastic (label `1`):

```text
I loovee when people text back ... 😒 #sarcastictweet
Don't you love it when your parents are Pissed because you were gonna
study after bubble soccer ! #IKnowIDo #not 😃 🔫
feeling like a million bucks after that chem 2 test . 😅 #not
```

Not sarcastic (label `0`):

```text
<user> Rest in peace & love to you and your family
Being half spanish and not being able to speak spanish is honestly so
disappointing
100 days until Christmas! 🌲 #too soon #not ready yet
```

The last non-sarcastic line contains the substring `#not` *inside* a
longer phrase (`#not ready yet`). The example tokenizer keeps `#not` as
its own tag if it is written that way; the original `TweetTokenizer`
does the same when `#not` is a standalone hashtag.

## What is not in the repo

| Artifact | Why it is missing |
| --- | --- |
| `glove.twitter.27B.200d.bin` | ~1–2 GB; the notebooks load it from the working directory |
| `glove_tt.txt` | Text-format GloVe used by `get_metrics_of_models.ipynb` |
| Full `variables/` trees for the SavedModels | Only `saved_model.pb` + `keras_metadata.pb` were uploaded |
| `baseline_models/svm_*.pkl`, `rf_*.pkl` | Referenced by the notebooks; only Decision Tree and GBT pickles are present |

`emoji2vec.bin` and `emoji2vec_twitter.bin` **are** in the repository
root. See [reproduction.md](reproduction.md) for how to fetch GloVe.
