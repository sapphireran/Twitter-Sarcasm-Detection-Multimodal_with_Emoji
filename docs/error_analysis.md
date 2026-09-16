# Error analysis notes

The 2023 notebooks score models and stop. They do not print misses. This
page uses `examples/10_hard_cases.py` and `examples/11_subtest_membership.py`
to describe *where* a cheap system fails, using only the CSVs.

## Three sarcastic buckets on test

A sarcastic test tweet is one of:

| Bucket | Count on test (1,000 sarcastic) | What a cheap system sees |
| --- | ---: | --- |
| Cue hashtag (`#not`, `#sarcasm`, `#yeahright`, …) | 614 | Free lunch; cue-rule precision is 1.0 |
| No cue, but at least one emoji | 37 | emoji2vec can fire; word order still needed for the flip |
| No cue, no emoji | 349 | Pooled emoji half is zeros; hashtag rule predicts 0 |

Run:

```bash
python3 examples/10_hard_cases.py --split test --limit 12
```

Typical hard rows (paraphrased from the printed list; the script quotes
the CSV verbatim):

- Polarity in the adjective, no markup: *“So many useless classes, great to be student.”*
- Thanks-for-the-opposite: *“Thank you, random guy, for sneaking up behind me…”*
- Self-disappointment without `#not`: *“Being half spanish and not being able to speak spanish is honestly so disappointing.”*

Those are exactly the cases mean-pooling smears (`great` + `useless` cancel)
and a two-layer BiLSTM can keep in order. That matches the 5-point test
gap between random forest (0.815) and BiLSTM+attention (0.864) on word-only
features in [experiments.md](experiments.md).

## Subtest is not a second world

`examples/11_subtest_membership.py` on this clone:

- 278/278 subtest texts appear verbatim in test (0 subtest-only rows)
- 276/276 emoji-bearing test tweets appear in subtest
- the two extra subtest rows are non-emoji test tweets
- subtest is 61.9% sarcastic vs 50% on test

Scoring subtest is “emoji-present test, plus two other test lines,” not
an independent population. Cite both splits. Do not treat 0.892 subtest
accuracy as a drop-in replacement for 0.874 test accuracy.

## Cue-rule ceiling

From `examples/03_sarcasm_cues.py` / `06_toy_baseline.py`, predict
sarcastic iff a cue hashtag is present:

| Split | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.624 | 0.975 | 0.196 | 0.327 |
| test | **0.807** | 1.000 | 0.614 | 0.761 |
| subtest | 0.863 | 1.000 | 0.779 | 0.876 |

On test that rule is within a point of the 2023 random forest (0.815) and
well above SVM (0.769). The BiLSTM+attention test accuracy (0.864 / 0.874)
is the first result that is clearly not the hashtag leak. Train still has
92 cue false positives; test has zero.

The four-feature nearest-centroid model in `06_toy_baseline.py` is *worse*
than the cue rule (test acc 0.406) because length is unscaled and swamps
the binary flags. That is left in the example on purpose: fusion without
scaling is a real failure mode, and it is the same shape as concatenating
a dense GloVe mean with a mostly-zero emoji mean.

Exact-set cues miss lookalikes. Hard-case #5 in the script output is
labeled sarcastic and contains `#notagain`, which is not `#not`.

## emoji2vec coverage

`examples/09_peek_emoji2vec.py` reads `emoji2vec_twitter.bin` (**1,661 × 200**,
mean L2 2.35) without gensim. Every fixture face (`😒😭😅😃😑😌👏`) is
present. On test:

- 276 tweets have an emoji token
- 258 of those have at least one table hit
- 423 token hits / 30 OOV emoji tokens

The 2023 `AverageVectorPerEmoji` path wrote zeros for those 30 OOV tokens
and for the 1,724 test tweets with no emoji. Multimodal concat is sparse
on the full test set; that is why the sklearn emoji deltas there are tiny.
