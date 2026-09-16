# Lexical baseline (regenerable)

`examples/lexical_baseline.py` trains a Laplace-smoothed multinomial Naive Bayes model on tweet tokens plus a few boolean cues. It does not use GloVe, emoji2vec, or Keras. Re-run:

```bash
python3 examples/lexical_baseline.py
python3 examples/lexical_baseline.py --strip-supervision-tags
```

Numbers below are from those two commands in this checkout (Python 3.12, NumPy 2.4).

## Supervision tags

The script treats these hashtags as gold-adjacent:

`#not`, `#sarcasm`, `#sarcastic`, `#sarcastictweet`, `#yeahright`, `#yeah_right`

613 of 2,000 test tweets carry at least one of them (30.6%). Every one of those 613 is labeled sarcastic, so a rule that predicts 1 iff a tag is present has precision 1.0 on test and subtest.

## Scores

`#tag` is that rule. `NB` is Naive Bayes. “Tags kept” includes the hashtags and `CUE:not_tag` / `CUE:sarcasm_tag`. “Tags stripped” deletes those hashtags from the text before featurization.

| Split | System | n | Acc | Prec | Rec | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | NB, tags kept | 39,780 | 0.844 | 0.827 | 0.839 | 0.833 |
| train | NB, tags stripped | 39,780 | 0.806 | 0.788 | 0.796 | 0.792 |
| train | #tag rule | 39,780 | 0.624 | 0.976 | 0.196 | 0.327 |
| test | NB, tags kept | 2,000 | 0.833 | 0.766 | 0.959 | 0.852 |
| test | NB, tags stripped | 2,000 | 0.744 | 0.695 | 0.869 | 0.772 |
| test | #tag rule | 2,000 | 0.807 | 1.000 | 0.613 | 0.760 |
| subtest | NB, tags kept | 278 | 0.871 | 0.833 | 0.988 | 0.904 |
| subtest | NB, tags stripped | 278 | 0.712 | 0.747 | 0.808 | 0.777 |
| subtest | #tag rule | 278 | 0.863 | 1.000 | 0.779 | 0.876 |

On the test set, keeping the tags is worth about **9 accuracy points** (0.833 vs 0.744). The tag rule alone already beats the stripped NB model on accuracy (0.807 vs 0.744) because the remaining sarcastic tweets are the hard ones: no `#not`, often no emoji.

The course-project BiLSTM + attention numbers (test acc 0.864 / 0.874) sit above the leaky NB model. That gap is the part of the work that actually needs GloVe / sequential context, not the part that reads `#not`.

## Tokens the model latches onto

With tags kept, the largest class-1 log-odds ratios are `#sarcastictweet`, `CUE:sarcasm_tag`, `#yeahright`, `#not`, and `CUE:not_tag`. After stripping, those disappear and the ranking shifts to topical leftovers (`#cannabis`, `#notcool`) and the ✋ emoji.

Class 0 keeps community / sincerity tags (`#whitepeopleproblems`, `#challengeaccepted`, `#rebel`, `#onelove`) in both settings.

## Errors the leaky model still makes

Test disagreements with tags kept are almost all **sarcastic tweets that never used a supervision hashtag**:

```text
gold=1 pred=0  Nothing like the excitement of going to an interview ... for a second job ...
gold=1 pred=0  I just love working at mcdonalds so much lol ... #work
```

After stripping, the model also misses obvious tagged jokes (`Holy crap I look great ... #not`, `I love watching golf ! #not`) because the only remaining words look sincere.

## How to read this next to the Keras table

| Claim | Supported by this baseline? |
| --- | --- |
| “A lot of the label is hashtag-visible” | Yes. Tag rule precision = 1.0 on test. |
| “Emoji2vec helps on the full test set” | Not tested here. Use the recorded Keras / sklearn deltas in [results.md](results.md). |
| “NB is a drop-in replacement for BiLSTM+ATT” | No. Different features, different capacity. |
