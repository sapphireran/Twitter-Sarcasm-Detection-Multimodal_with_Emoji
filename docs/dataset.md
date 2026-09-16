# Dataset notes

Personal notes for the 2023 UCPH Computational Cognitive Science 2 project. The
checked-in CSVs are the only data the new example scripts need.

## Files

| File | Rows | Role |
| --- | ---: | --- |
| `dataset/train_sentence.csv` | 39,780 | Training tweets |
| `dataset/train_label.csv` | 39,780 | Training labels |
| `dataset/test_sentence.csv` | 2,000 | Official test tweets |
| `dataset/test_label.csv` | 2,000 | Official test labels |
| `dataset/subtest_sentence.csv` | 278 | Emoji-bearing slice of test |
| `dataset/subtest_label.csv` | 278 | Matching subtest labels |

Each sentence file is one tweet per line. Some lines are quoted because they
contain commas. Each label file is a single `0` or `1` per line, no header.

- `0` = not sarcastic
- `1` = sarcastic

`data_utils.ReadOpen` does **not** parse the files as real CSV. It reads raw
lines, then replaces commas with spaces:

```python
sentence = " ".join(line.strip().split(","))
```

That is why the example tokenizer starts with the same comma rewrite.

## Label balance

| Split | Non-sarcastic (`0`) | Sarcastic (`1`) | Sarcastic % |
| --- | ---: | ---: | ---: |
| train | 21,292 | 18,488 | 46.5% |
| test | 1,000 | 1,000 | 50.0% |
| subtest | 106 | 172 | 61.9% |

Train is close to balanced. Test was constructed as an even split. Subtest is
skewed toward sarcastic tweets.

## What subtest actually is

Every subtest sentence also appears in the official test file. Subtest is the
subset of test tweets that contain emoji (278 / 2,000). That is the split the
2023 notebooks used to ask: *does emoji2vec help when emoji are actually
present?*

Mean length is similar across splits (about 16–18 whitespace tokens). Subtest
tweets are slightly longer and much richer in hashtags.

## Emoji and hashtag rates

Computed from the CSVs with `python -m examples.dataset_report` (emoji =
codepoints matched by `examples.lib.tweet_features.EMOJI_RE`):

| Split | Mean tokens | With emoji | With a hashtag |
| --- | ---: | ---: | ---: |
| train | 18.4 | 13.8% | 21.3% |
| test | 16.9 | 13.8% | 46.6% |
| subtest | 17.9 | 99.6% | 55.8% |

The official test set is not a random draw from train; it is denser in
explicit sarcasm markers. A live copy of the same tables lives in
[generated/dataset_snapshot.md](generated/dataset_snapshot.md).

## Cue shift between train and test

Substring rates on lowercase text:

| Cue | Train y=0 | Train y=1 | Test y=0 | Test y=1 | Subtest y=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `#not` | 0.6% | 18.2% | 0.2% | 51.3% | 67.4% |
| `#sarcasm` | ~0% | ~0% | 0% | 8.9% | 0% |
| `#sarcastictweet` | 0% | 1.2% | 0% | 3.8% | 10.5% |
| `love` | 11.5% | 13.6% | 4.3% | 34.6% | 50.6% |

`#not` is a strong sarcasm cue everywhere, but it is *much* stronger on the
evaluation sets than on train. A model that only memorizes `#not` will look
better on test than its train fit suggests. The lexical baseline example is
there to make that visible.

The tweet `100 days until Christmas! 🌲 #too soon #not ready yet` is labeled
`0`. Here `#not` is part of “not ready yet”, not a sarcasm hashtag. Surface
cues are useful and noisy.

## Frequent emoji

Train, both classes, is dominated by 😂, ❤, 😍, 😭, 😊. Those faces are
common emotional punctuation, not reliable flips.

On the official test set the sarcastic class leans toward 😒, 😊, 😅, 🔫, 😑.
The non-sarcastic class still likes 😂 and 😭. `python -m examples.emoji_signals`
prints smoothed log-odds if you want the full ranking.

## Overlap

- Unique train strings: 39,758 / 39,780 (a few exact duplicates)
- Unique test strings: 2,000 / 2,000
- Exact train ∩ test string overlap: 48 tweets

The 48 overlapping strings are a small leak. The examples do not drop them,
because the original notebooks did not either.

## Embeddings that are *not* in the CSV folder

The 2023 notebooks also used:

| Artifact | In this clone? | Notes |
| --- | --- | --- |
| `glove.twitter.27B.200d.bin` | no | GloVe Twitter 200-d, word2vec binary |
| `emoji2vec_twitter.bin` | yes | ~1.3 MB, used for the multi-modal path |
| `emoji2vec.bin` | yes | ~2.0 MB, generic emoji2vec |
| `baseline_models/*.pkl` | no | directory is empty in this clone |
| `model/best_model_*` | incomplete | only `keras_metadata.pb` remains |

The new `examples/` scripts never load those binaries. They only read the CSVs.

## How to regenerate the snapshot

```bash
python3 -m examples.dataset_report --markdown -o docs/generated/dataset_snapshot.md
```
