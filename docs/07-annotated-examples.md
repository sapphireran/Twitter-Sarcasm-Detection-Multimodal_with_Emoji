# 07 — Annotated examples

Hand-read rows from the shipped CSVs. Labels are the file labels, not a
fresh annotation. Quotes are truncated where the tweet is long. `<user>`
is already a placeholder in the export.

## Subtest (every row has an emoji) — sarcastic

These are the textbook multimodal cases: positive predicate, deadpan or
hyperbolic face, often a cue hashtag.

| Label | Tweet | Why it is (labeled) sarcastic |
| --- | --- | --- |
| 1 | I loovee when people text back ... 😒 `#sarcastictweet` | Elongated “love” + unamused face + explicit tag. Word channel sees “love”; emoji channel should cancel it. |
| 1 | Don't you love it when your parents are Pissed ... `#IKnowIDo` `#not` 😃 🔫 | “Love” + smile + gun is a classic mismatch. `#not` is a leaked label. |
| 1 | I just love having grungy ass hair 😑 `#not` | “Love” + expressionless face. |
| 1 | feeling like a million bucks after that chem 2 test . 😅 `#not` | Conventional irony; `#not` again. |

A model that only averages GloVe will still often get these right because
`#not` / `#sarcastictweet` survive tokenization. The WE vs W comparison
is only informative if those tags are *also* in the W model — they are.
The extra signal is the face when the tag is missing or when attention
can use both.

## Subtest — literal

| Label | Tweet | Notes |
| --- | --- | --- |
| 0 | I cannot speak to people that are so self indulged 😴 shutupp | Face is tired / done, aligned with the text. Not a polarity flip. |
| 0 | i dont have the confident to speak ... 😭 😭 😭 | Affect matches content. |
| 0 | He really pissed me off last night ... 😒 | Unamused face is sincere here. Same pictograph as the sarcastic “love … 😒” row — **context** has to disambiguate. |
| 0 | There is this 1 quince picture ... 😂 | Laughing face as social smoothing, not irony. |

The last two rows are why a unigram “😒 → sarcastic” rule fails.
`examples/02_emoji_cooccurrence.py` will show 😒 enriched for class 1, but
not exclusively.

## Train — sarcastic with `#not`

| Label | Tweet |
| --- | --- |
| 1 | So nerveous! Love a good bidding war 😁 `#NOT` |
| 1 | Sitting by myself for lunch is cool... `#not` 😔 |
| 1 | "I seriously LOVEEEE when people lie to me, makes me feel special `#not` 😒 " |
| 1 | I love how hypocritical he is 😂 `#not` |

Cue-tag leakage is obvious. Any honest ablation should strip `#not` /
`#sarcasm*` and re-score. The 2023 notebooks did not.

## Train — sarcastic, no hashtag

These are the rows a sequence model is *for*. Some look correctly ironic;
some look like label noise (distant supervision leftovers, or a
mismatched file join).

| Label | Tweet | Comment |
| --- | --- | --- |
| 1 | Imagine how awesome life would be if you could be allergic to homework. | Conventional counterfactual irony. Clean. |
| 1 | I don't stalk. I just update myself on what's going on in people's lives. | Self-cancelling definition. Good sarcasm. |
| 1 | `<user>` just caught this on Watch TCM! This was a fun flick! The chemistry between Jimmy Cagney and Joan Blondell was perfect | Reads as sincere praise. **Possible mislabel.** |
| 1 | ive been tryna sleep for like a half hour now and im just so stressed about a coworker i cant even sleep. | Reads as a sincere complaint. **Possible mislabel.** |

If you retrain, consider a tiny cleaned probe set of 50 hand-confirmed
rows and reporting that next to test/subtest. The 2023 table cannot
tell you how much of 87% is cue tags plus noise.

## Train — literal, with emoji

| Label | Tweet | Comment |
| --- | --- | --- |
| 0 | My long luscious hair is gone. 😢 😢 😢 😢 | Sincere disappointment. |
| 0 | Love being home with a full fridge 😋 | Positive predicate, matching face. Contrast with “love … 😒”. |
| 0 | Happy birthday baby `<user>` ... ❤ ❤ | Affectionate; ❤ is a literal-leaning pictograph in the PMI table. |
| 0 | 100 days until Christmas! 🌲 `#too` soon `#not` ready yet | Literal `#not` (“not ready”). This is the false-positive mode of a hashtag rule. |

## What a walkthrough of the net *should* show

On “I loovee when people text back ... 😒 `#sarcastictweet`”:

1. Tokenizer keeps `loovee`, `😒`, `#sarcastictweet`.
2. GloVe may miss `loovee` (elongation) unless a fuzzy path exists; the
   2023 code does not normalize elongations. The face and the hashtag
   carry the example.
3. Attention (after training) should put mass on `#sarcastictweet` and
   `😒`. `examples/05_toy_attention.py` only shows the *mechanism*, not
   these weights — recovering them would need the missing SavedModel
   shards.

On “Love being home with a full fridge 😋”:

1. `love` and `😋` should agree in sign inside emoji2vec-near-GloVe space.
2. WE should *not* flip the label relative to W.

That pair is the qualitative version of the subtest metric.
