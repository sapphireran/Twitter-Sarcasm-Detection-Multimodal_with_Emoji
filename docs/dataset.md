# Dataset

The corpus is a binary sarcasm collection of tweets that already include
`<user>` placeholders. Labels are `0` (non-sarcastic) and `1` (sarcastic).
Each split is a pair of line-aligned files under `dataset/`.

| Split | Sentences | Labels | Sarcastic | Rate | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| `train` | 39,780 | 39,780 | 18,488 | 0.465 | Fit embeddings-side models |
| `test` | 2,000 | 2,000 | 1,000 | 0.500 | Official in-domain test |
| `subtest` | 278 | 278 | 172 | 0.619 | Emoji-bearing slice of `test` |

Those counts come from `examples/inspect_dataset.py`, not from the original
README (which only named the course). Re-run that script after any corpus
edit.

## File format

`*_label.csv` is one integer per line. `*_sentence.csv` is one tweet per line.
Tweets that contain commas are wrapped in CSV quotes:

```
"Late nights, early mornings is how I live my life."
```

`data_utils.ReadOpen` does not use a CSV parser. It rebuilds each line as
`' '.join(line.strip().split(','))`, so an interior comma becomes a space.
`sarcasm_lib.io.read_sentences` keeps the comma. The two readers are compared
in `examples/tokenize_tweets.py`.

## Subtest is not a held-out domain

Every subtest tweet is also in the official test file. 276 / 278 tweets
contain pictograph emoji; the remaining two use miscellaneous symbols that
the lightweight extractor currently misses (`⭕` and a regional-indicator
flag written with spaces). Treat subtest as "test tweets that carry emoji",
not as a second population.

That construction explains three facts at once:

1. Subtest is class-imbalanced (172 / 106) even though test is balanced.
2. Mean emoji per tweet jumps from 0.23 on test to 1.63 on subtest.
3. Every trained model, and the hashtag heuristic, scores higher on subtest
   than on the full test set.

## Surface cues

Hashtags are the loudest label leak in this corpus.

| Marker | Train sarcastic / non-sarcastic |
| --- | --- |
| `#not` | 3,364 / 127 |
| `#sarcastic` | 283 / 4 |
| `#yeahright` | 237 / 4 |
| `#sarcastictweet` | 228 / 1 |

On the official test set the union of the marker set in
`sarcasm_lib.heuristic.SARCASM_HASHTAGS` has precision 1.0 (616 tweets, all
sarcastic). `#sarcasm` itself is absent from train and appears 89 times on
test — a split mismatch worth remembering if you retrain a hashtag feature.

Train still has plenty of sarcastic tweets *without* those markers. That is
why a cue-only heuristic only reaches 0.20 recall on train and 0.62 recall
on test. The Bi-LSTM is doing work that `#not` cannot do.

## Emoji

Emoji are common but not a sarcasm detector by themselves.

| Split | Tweets with emoji | Of those, sarcastic | Of those, non-sarcastic |
| --- | ---: | ---: | ---: |
| train | 5,458 | 2,273 | 3,185 |
| test | 276 | 171 | 105 |
| subtest | 276 | 171 | 105 |

On train, emoji are slightly *more* common in non-sarcastic tweets. On the
emoji slice they correlate with sarcasm because that slice was selected for
emoji, not because a face glyph is a reliable cue. The useful signal is the
*clash*: a cheerful opener (`I just love…`) next to `😒` / `😑` / `😩`.

Most frequent glyphs on train: 😂, 😊, 😭, 😍, 😒, ❤️, 😩, 😘, ❤, 😁.

## Typical length

Mean token count after `sarcasm_lib.tokenize.tokenize_tweet` is about 18 on
train and subtest, 17 on test. The saved Keras models pad to length 78, which
is the maximum training sequence from the original `Preprocess` run.

## Examples

Non-sarcastic (train):

```
<user> Rest in peace & love to you and your family
```

Sarcastic, marker hashtag (subtest):

```
I loovee when people text back ... 😒 #sarcastictweet
```

Sarcastic, polarity clash without `#not` (train):

```
Expecting is my favorite crime and disappointment is always my punishment.. 😎😎 😥 😊
```

The inspect script prints more of these, plus the top hashtags and emoji for
every split.
