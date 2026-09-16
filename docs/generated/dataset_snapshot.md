# Dataset snapshot

Computed live from `dataset/*.csv`. Label `1` is sarcastic; label `0` is not sarcastic.

| Split | Tweets | Sarcastic | Non-sarcastic | Mean tokens | With emoji | With hashtag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 39780 | 18488 | 21292 | 18.4 |  13.8% |  21.3% |
| test | 2000 | 1000 | 1000 | 16.9 |  13.8% |  46.6% |
| subtest | 278 | 172 | 106 | 17.9 |  99.6% |  55.8% |

## Cue rates by class

A cue is counted when the lowercase tweet contains the substring.

| Split | Class | n | #not | #sarcasm | #sarcastictweet | love | great |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | sarcastic | 18488 |  18.2% |   0.0% |   1.2% |  13.6% |   4.1% |
| train | non-sarcastic | 21292 |   0.6% |   0.0% |   0.0% |  11.5% |   4.7% |
| test | sarcastic | 1000 |  51.3% |   8.9% |   3.8% |  34.6% |   5.7% |
| test | non-sarcastic | 1000 |   0.2% |   0.0% |   0.0% |   4.3% |   3.2% |
| subtest | sarcastic | 172 |  67.4% |   0.0% |  10.5% |  50.6% |   5.2% |
| subtest | non-sarcastic | 106 |   0.0% |   0.0% |   0.0% |   4.7% |   0.9% |

## Split overlap

- unique train tweets: 39758
- unique test tweets: 2000
- subtest tweets also in test: 278 / 278
- train/test exact string overlap: 48

Subtest is the emoji-bearing slice of the official test set. That is why multi-modal gains show up more clearly there.

## Explicit sarcasm hashtags

- train: 3480 tweets contain ['#ironic', '#irony', '#not', '#sarcasm', '#sarcastic', '#sarcastictweet']
- test: 595 tweets contain ['#ironic', '#irony', '#not', '#sarcasm', '#sarcastic', '#sarcastictweet']
- subtest: 129 tweets contain ['#ironic', '#irony', '#not', '#sarcasm', '#sarcastic', '#sarcastictweet']
