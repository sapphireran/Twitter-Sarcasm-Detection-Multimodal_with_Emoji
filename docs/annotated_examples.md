# Annotated test tweets

All rows below are copied from `dataset/test_sentence.csv` with the
parallel label from `dataset/test_label.csv`. They are grouped by
the cue that makes the label easy or hard. None of these are
synthetic.

## Explicit sarcasm hashtag

Label `1`. The tag is the label.

```
I loovee when people text back ... 😒 #sarcastictweet
```

Cues stacked on one line: elongated `loovee` (positive surface),
ellipsis, unamused emoji, then `#sarcastictweet`. A bag-of-means
model has to average `love` with `😒`; attention can put mass on
the tag and the emoji.

```
I love walking to school 😄 #SarcasticTweet
```

Positive verb + smiling emoji + tag. The emoji polarity does **not**
match the label; the hashtag does. emoji2vec can hurt this row if
the network trusts 😄 as sincere.

## `#not` after a positive frame

Label `1`. This is the dominant sarcastic pattern on the test split
(465 rows contain `#not`).

```
Oh how I love getting home from work at 3am and my house being dirty #not
```

```
I just love having grungy ass hair 😑 #not
```

```
feeling like a million bucks after that chem 2 test . 😅 #not
```

```
Don't you love it when your parents are Pissed because you were gonna study after bubble soccer ! #IKnowIDo #not 😃 🔫
```

The last one is a pile-up: rhetorical `Don't you love`, anger
(`Pissed`), `#IKnowIDo`, `#not`, grin, pistol. Classical concat
throws 😃 and 🔫 into one emoji mean; the LSTM keeps their order.

## Elongation and school-register sarcasm

Label `1`.

```
Good thing I spent four years getting MLA formatting shoved down my throat . #itsAPA 😑 👎
```

No `#not`. The flip is lexical (`Good thing` + complaint) plus
`#itsAPA` and two negative emoji. A hashtag-only rule misses it.

```
Anyone in Chem that has to do OWLs knows my pain right now ... #crazycollgelife #not #studying24hrsaday 😩
```

Typo hashtag `#crazycollgelife` still tokenizes as one `#` token
under `TweetTokenizer`.

## Non-sarcastic tweets that still have emoji

Label `0`. These are why “has emoji ⇒ sarcastic” is a bad rule.
The test set has 277 emoji rows and only 1,000 sarcastic rows.

```
Want to have someone to speak to I'm so bored 😭
```

```
So annoyed about everything and now I have to go to class and give a speech sick af and can barley speak 😅 😅
```

```
"I feel so honored 2 organizations , one from NCCU & one from UNCG , want me to come speak at their school 😌"
```

```
follow me so you can actually speak to me instead of writing bitchy tweets and then deleting them 👏 👏 👏
```

Emoji here mark stance or emphasis, not a polarity flip. The
subtest is full of this class of row mixed with the sarcastic
emoji rows — that is the actual multi-modal question.

## Literal, no emoji, no sarcasm tag

Label `0`.

```
i just imagined you dancing like this
```

A mean GloVe vector of common words; no `#not` to latch onto. The
deep model has to stay below 0.5 on ordinary timeline text or the
false-positive rate explodes.

## Why the lexical example exists

`examples/lexical_sarcasm_baseline.py` turns the cues in this file
into a feature vector:

- `has_not_hashtag`, `has_sarcasm_hashtag`, `has_yeahright`
- emoji count, elongated-token count
- `!` / `?` / `...` counts
- a coarse “positive word + complaint” flag

It will look strong on the `#not` slice and weak on the MLA /
honored-to-speak rows. That gap is the reason the coursework
bothered with LSTMs and emoji2vec at all.
