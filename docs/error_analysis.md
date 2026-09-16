# Error analysis notes

The 2023 notebooks score models and stop. They do not print misses. This
page uses `examples/10_hard_cases.py` and `examples/11_subtest_membership.py`
to describe *where* a cheap system fails, using only the CSVs.

## Three sarcastic buckets on test

A sarcastic test tweet is one of:

| Bucket | Count (recomputed) | What a cheap system sees |
| --- | ---: | --- |
| Cue hashtag (`#not`, `#sarcasm`, …) | from `10_hard_cases.py` | Almost a free lunch; precision of the cue rule is 1.0 on test |
| No cue, but at least one emoji | same script | emoji2vec can fire; word order still needed for the flip |
| No cue, no emoji | same script | Pooled emoji half is a zero vector; hashtag rule predicts 0 |

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

`examples/11_subtest_membership.py` checks set overlap:

- Almost every emoji-bearing *test* tweet appears in *subtest*.
- Subtest is slightly larger (278 vs 266 emoji test tweets) and more
  sarcastic (61.9% vs 50%).
- Scoring subtest is therefore “how do we do on the emoji-present slice
  of test, plus a few extras,” not an independent population.

Cite both splits, or cite emoji-present vs emoji-absent *inside* test.
Do not treat 0.892 subtest accuracy as a drop-in replacement for 0.874
test accuracy.

## Cue-rule ceiling

From `examples/03_sarcasm_cues.py`, predicting sarcastic iff a cue
hashtag is present:

- **Test:** precision 1.0, recall is the cue rate among sarcastic tweets
  (595 / 1000 if every cue tweet is sarcastic — the script prints the
  live TP/FN).
- **Train:** a few false positives (cue hashtag on a `0` label). The
  distant-supervision story leaked into train more messily than into test.

Any neural number you quote should be compared to this rule, not only to
chance (0.50 on test).

## emoji2vec coverage

`examples/09_peek_emoji2vec.py` reads `emoji2vec_twitter.bin` (1,661 × 200)
without gensim and counts test-set hits. If a face emoji used in the
fixture is missing, the 2023 `AverageVectorPerEmoji` path wrote zeros for
that token. Coverage is part of the multimodal claim; it is not 100%.
