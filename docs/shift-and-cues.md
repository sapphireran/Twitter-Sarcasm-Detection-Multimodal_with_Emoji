# Distribution shift and cue leakage

The official test set is not a matched draw from train. Two shifts
explain most of the “models look better on subtest” pattern.

## 1. Subtest is the emoji slice of test

Every unique subtest string appears in test. None appear in train.
Emoji coverage:

| Split | Rows with an emoji | Share |
| --- | ---: | ---: |
| train | 5,458 | 13.7% |
| test | 276 | 13.8% |
| subtest | 276 | 99.3% |

Test and train have almost the same emoji rate. Subtest was *cut from
test* by that filter. A WE model is therefore being scored on the only
slice where the extra 200d emoji channel can fire.

## 2. Test is cue-heavier than train

Explicit sarcasm hashtags after lowercasing and tweet tokenization:

| Cue | Train n | Train sarcasm rate | Train coverage | Test n | Test sarcasm rate | Test coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `#not` | 3,491 | 0.964 | 0.088 | 515 | 0.996 | **0.258** |
| `#sarcasm` | 0 | — | 0 | 89 | 1.000 | 0.044 |
| `#sarcastic` | 287 | 0.986 | 0.007 | 42 | 1.000 | 0.021 |
| `#sarcastictweet` | 229 | 0.996 | 0.006 | 38 | 1.000 | 0.019 |
| `#yeahright` | 241 | high | 0.006 | 19 | high | 0.010 |

`#not` is three times more common on test than on train, and it is
almost a perfect label. `#sarcasm` does not appear in train at all,
so any test tweet that uses it is an out-of-vocabulary hashtag the
2023 embedding table could only handle as a zero or as a hashed
unknown.

A rule that predicts sarcastic iff an explicit cue is present is
therefore a serious baseline on test and a weak one on the cue-free
remainder. `examples/05_cue_rule.py` measures that split.

## 3. Why WE gains grow on subtest

Recorded BiLSTM + attention accuracy:

* test: W 0.8635 → WE 0.8735 (+1.0)
* subtest: W 0.8669 → WE 0.8921 (+2.5)

Random forest shows the same shape (test +0.4, subtest +4.7). SVM
actually *loses* 0.6 on test WE and gains 1.1 on subtest WE. Averaged
emoji vectors are a blunt instrument: they help when an emoji is
present and can add noise when the emoji channel is all zeros.

## 4. Classroom takeaway

If you only quote the official test accuracy, you mix three effects:

1. genuine word-sequence sarcasm
2. hashtag leakage (`#not`, `#sarcasm`, …)
3. emoji-channel signal on 14% of the rows

The archive lab reports slice metrics (`explicit_cue` vs
`no_explicit_cue`, `has_emoji` vs `no_emoji`) so those effects stay
separated.
