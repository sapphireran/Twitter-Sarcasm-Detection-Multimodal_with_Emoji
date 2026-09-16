# Example script output

Snapshot from this branch, run on the committed `dataset/` files (Python 3.12, NumPy 2.4). Re-run the commands if the tokenizer or cue list changes.

## `python3 examples/dataset_overview.py`

```text
Splits (labels 0 = not sarcastic, 1 = sarcastic)
split        n  label0  label1  sarcasm  mean tok  p50  p95  max
-------  -----  ------  ------  -------  --------  ---  ---  ---
train    39780   21292   18488    46.5%      17.9   17   30   90
test      2000    1000    1000    50.0%      16.5   16   29   38
subtest    278     106     172    61.9%      17.6   17   30   34

How often a tweet contains at least one hashtag / emoji / <user>
split    hashtag   emoji  mention  mean chars
-------  -------  ------  -------  ----------
train      21.3%   13.8%    23.7%        84.3
test       46.6%   13.9%     0.0%        79.0
subtest    55.8%  100.0%     0.0%        80.9
```

This tokenizer’s train max length is 90 (NLTK / Keras used 78).

## `python3 examples/cue_analysis.py`

```text
Share of tweets with each cue (overall / not-sarcastic / sarcastic)
split    cue                all  label0  label1
train    sarcasm hashtag   9.4%    0.4%   19.7%
train    neg. emoji        3.7%    3.2%    4.2%
train    positive stem     9.7%    8.7%   10.9%
test     sarcasm hashtag  30.8%    0.0%   61.5%
test     neg. emoji        5.7%    2.3%    9.1%
test     positive stem    20.1%    4.5%   35.6%
subtest  sarcasm hashtag  48.2%    0.0%   77.9%
subtest  neg. emoji       41.0%   21.7%   52.9%
subtest  positive stem    34.9%    4.7%   53.5%

Most common training hashtags
#not 3188 (97.4% sarcastic)
#yeahright 241 (98.3%)
#sarcastictweet 229 (99.6%)
```

## `python3 examples/lexical_baseline.py`

Vocabulary: 12,964 tokens (`min_df=2`, max 15,000).

| model | n | acc% | p% | r% | f1% |
| --- | ---: | ---: | ---: | ---: | ---: |
| NB / test | 2000 | 84.75 | 79.42 | 93.80 | 86.02 |
| NB / subtest | 278 | 85.61 | 82.67 | 97.09 | 89.30 |
| SGD-logreg / test | 2000 | 87.95 | 90.59 | 84.70 | 87.55 |
| SGD-logreg / subtest | 278 | 89.93 | 92.35 | 91.28 | 91.81 |
| rule-cues / test | 2000 | 81.30 | 99.53 | 62.90 | 77.08 |
| rule-cues / subtest | 278 | 88.85 | 100.00 | 81.98 | 90.10 |
| count-cues / test | 2000 | 80.25 | 96.90 | 62.50 | 75.99 |
| count-cues / subtest | 278 | 82.73 | 87.80 | 83.72 | 85.71 |

NB train accuracy 84.23% (not wildly overfit). Rule-cues train accuracy 62.53% — the distant-supervision tags are much denser on test/subtest than on train.

Interpretation: [findings.md](findings.md).

## `python3 examples/error_analysis.py`

```text
Confusion counts (TP FP TN FN)
rule / test     629    3  997  371
rule / subtest  141    0  106   31
NB / test       938  243  757   62

Sarcastic test tweets with a lexicon sarcasm hashtag: 615 / 1000 (61.5%)
```

Rule false negatives are verbal irony without `#not` (“So many useless classes , great to be student”). Rule false positives are rare (3 on the 2,000-tweet test set).

## `python3 examples/token_odds.py`

Strongest sarcastic log-odds include `#sarcastictweet` (4.79), `#yeahright` (3.92), `#not` (3.66), `#sarcastic` (2.69). Strongest non-sarcastic tags include `#whitepeopleproblems` and `#challengeaccepted`. The long tail (`#cannabis`, `clinicals`) is low-support and should not be over-read.

## `python3 examples/attention_walkthrough.py`

On the synthetic “I love walking to `<pad>` `<pad>` school `#not`” sequence, masked attention puts ~0.30 on `school` and ~0.30 on `#not`, zeros the pads, and sums to 1. Unmasked pads take ~0.08 each and move the context (L2 ≈ 0.20).
