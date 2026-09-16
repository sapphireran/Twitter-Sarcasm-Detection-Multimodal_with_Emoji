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
| any explicit tag | 3,733 | 0.975 | 0.094 | 615 | 1.000 | **0.308** |
| `#not` token | 3,202 | 0.974 | 0.080 | 467 | 1.000 | **0.234** |
| other sarcasm tags | 532 | 0.983 | 0.013 | 150 | 1.000 | 0.075 |

Counts are **token** matches after `tokenize_tweet`, not substring
search. A raw `#not in line` count is higher because it also hits
`#notes` / `#nothing`. Full Wilson intervals are in
`docs/generated/cue_shift.md`.

Explicit cues are about **3× more common on test than on train**, and
every test hit is labeled sarcastic. A rule that predicts sarcastic
iff an explicit tag is present scores **0.8075 accuracy / 1.000
precision / 0.615 recall** on official test — **above the recorded
SVM W accuracy of 0.769**. On the 1,385 test tweets with no explicit
tag, that rule’s recall is zero by construction. That remainder is
what a sequence model still has to earn.

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

A hashed logistic fit on a **stratified** 12,000-row train sample
(`docs/generated/hashed_baseline.md`) reached 0.773 test accuracy
(recorded SVM W: 0.769). Cue weights put `explicit` and `not_tag`
first. The same model is perfect on the 615 test tweets with an
explicit cue and falls to 0.672 accuracy / 0.371 F1 on the other
1,385. A raw prefix of train cannot be used for that experiment:
the first 12,000 train rows are 11,991 sincere.
