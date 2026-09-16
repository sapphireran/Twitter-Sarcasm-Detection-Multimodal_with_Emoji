# Dataset

Three CSV pairs live in `dataset/`. Each tweet is one line in
`*_sentence.csv`; the matching 0/1 label is the same line in
`*_label.csv`. `1` is sarcastic.

| Split | Rows | Sarcastic | Sincere | Mean tokens | Max tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 39,780 | 18,488 (46.5%) | 21,292 (53.5%) | 16.5 | 51 |
| test | 2,000 | 1,000 (50.0%) | 1,000 (50.0%) | 16.5 | 36 |
| subtest | 278 | 172 (61.9%) | 106 (38.1%) | 17.7 | 36 |

Counts come from `examples/01_split_census.py` on this clone.

## Integrity

`ccs2lab.splits.integrity` checks the facts the rest of the docs use:

| Check | Result |
| --- | --- |
| sentence / label alignment | 39,780 / 2,000 / 278 |
| labels only in `{0, 1}` | yes |
| unique train ∩ test | 48 tweets |
| unique train ∩ subtest | 0 |
| unique test ∩ subtest | 278 |
| subtest ⊆ test | **yes** |

Subtest is therefore **the emoji-bearing subset of the official test
set**, not a third draw from the same stream. 276 / 278 subtest rows
contain at least one emoji character (99.3%). The two rows without a
detected emoji still survived the original “has emoji” filter — they
are almost certainly symbol-only leftovers.

The 48 train/test string collisions are leftover duplicates, not a
designed leak. They are 2.4% of test. The archive lab reports them
and does not drop them, because the 2023 notebooks did not drop them
either.

## How the CSVs are stored

`ReadOpen` in `data_utils.py` does not use a CSV parser. It reads
raw lines, strips them, and joins on commas:

```python
sentence = ' '.join(line.strip().split(','))
```

Quoted tweets that contain commas therefore lose the comma and keep
the words. Mentions were already normalized to `<user>` in some rows;
others still have `@`. Hashtags are kept as surface text (`#not`,
`#sarcastictweet`).

## Label source

The files look like a Twitter sarcasm collection that mixes:

* hashtag-labeled sarcastic posts (`#not`, `#sarcasm`, `#sarcastictweet`)
* hashtag-labeled or distant-supervised sincere posts
* a smaller amount of emoji-only contrast

Distant supervision from sarcasm hashtags is the usual construction
for this genre (see Riloff et al. and later Twitter sarcasm corpora).
That construction is also the main evaluation hazard: a model can
score well by reading the hashtag that *defined* the label.

## Examples (from train)

Sarcastic:

* `Being sore is the best and the worst feeling in the world`
* `Expecting is my favorite crime and disappointment is always my punishment.. 😎😎 😥 😊`
* `Happy birthday to me. Yay.`

Sincere:

* `<user> i hope youre lurking rn. i want to listen to hallucination & wanna love you again live someday, pretty please?! 😭 😭 😭`
* `100 days until Christmas! 🌲 #too soon #not ready yet`

The last sincere row is why `#not` is a noisy cue: the token is
present, the label is 0.
