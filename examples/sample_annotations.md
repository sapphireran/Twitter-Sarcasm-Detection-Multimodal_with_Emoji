# Annotated sample tweets

Hand-picked rows from the official files, with a short note on *why* a multi-modal model should or should not need the emoji channel. Labels are the file labels, not a new annotation.

## Test set — sarcastic, emoji + wording clash

**`test[0]` gold=`1`**

```text
I loovee when people text back ... 😒 #sarcastictweet
```

- Positive predicate (`loovee`) + unamused face + explicit tag.
- `😒` is in `emoji2vec_twitter.bin` (nearest: 😞, 🙎, 🙁). GloVe will miss it.
- A mean-pool model can add the 😒 vector to the 400-d concat. A Bi-LSTM can sit `loovee` and `😒` on neighbouring steps.
- Cue-tag baseline also gets this one for free (`#sarcastictweet`).

**`test[1]` gold=`1`**

```text
Don't you love it when your parents are Pissed because you were gonna study after bubble soccer ! #IKnowIDo #not 😃 🔫
```

- Rhetorical `love` + `Pissed` + `#not` + smile + pistol.
- The smile-plus-pistol pair is itself a clash; emoji2vec places 🔫 near 💥 / 🎉 in the Twitter table (noisy) and 😃 near 😊 / 😀.
- Attention *should* be allowed to land on `#not` and `Pissed` even if the emoji rows are messy.

**`test[3]` gold=`1`**

```text
Oh how I love getting home from work at 3am and my house being dirty #not
```

- No emoji. Single-modal and multi-modal deep models see the same GloVe rows here (`get_emoji2vec` only fills OOV / emoji tokens).
- This is the example that explains why the *test-set* emoji gain is only +1.0 acc: most sarcastic tweets in `test` do not need the emoji table.

## Test set — sarcastic, no cheap hashtag

**`test[2]` gold=`1`**

```text
"So many useless classes , great to be student"
```

- After comma collapse: `so many useless classes   great to be student`.
- Clash is lexical (`useless` vs `great`), not emoji and not a sarcasm hashtag.
- Cue-tag baseline **misses** this. Clash baseline should catch it. This is the kind of row a Bi-LSTM has to win on if it wants credit beyond distant supervision.

## Train set — non-sarcastic, emoji present

**`train[0]` gold=`0`**

```text
<user> i hope youre lurking rn. i want to listen to hallucination & wanna love you again live someday, pretty please?! 😭 😭 😭
```

- Three 😭 tokens. emoji2vec neighbours include 😿 / 😢 / 😂 — mixed crying / laughing.
- Gold is not sarcastic: the tears read as sincere pleading.
- A model that treats “any emoji ⇒ sarcastic” will false-positive here. The subtest construction (almost every row has an emoji, *both* labels) is what stops that cheat.

**`train[4]` gold=`0`**

```text
100 days until Christmas! 🌲 #too soon #not ready yet
```

- Contains `#not`, but the tag is part of the phrase `#not ready yet`, and the gold label is `0`.
- This is the 91-row train leak the dataset page mentions: cue tags are not ground truth.
- Tokenization must keep `#not` as a hashtag and `🌲` as a token (nearest: 🎄, 🌳).

## Subtest — why the slice exists

Subtest is not “a smaller test set”. It is the 278 rows where emoji co-occurrence is almost always present (277 / 278). Gains from `emoji2vec_twitter.bin` are supposed to show up *here*:

| Model (2023) | text acc | +emoji acc | Δ |
| --- | ---: | ---: | ---: |
| Random forest | 0.806 | 0.853 | +0.047 |
| Bi-LSTM + Att | 0.867 | 0.892 | +0.025 |

If you add a new fusion method, report this slice before you claim “emoji helps”.

## How to print more rows

```bash
python examples/tokenize_tweets.py --split subtest
python examples/cue_baseline.py --errors 8 --error-rule cue-tag --split test
```
