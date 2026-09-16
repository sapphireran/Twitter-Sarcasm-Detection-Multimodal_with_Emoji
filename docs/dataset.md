# Dataset

The course project uses a Twitter sarcasm corpus stored as line-aligned
CSV pairs under `dataset/`. Labels are `0` (literal) and `1` (sarcastic).
Each sentence file has one tweet per line; some lines are wrapped in
quotes because the text itself contains commas.

These files already live in the repository. The examples never fetch new
posts from Twitter/X.

## Splits

| Split | File pair | Rows | Sarcastic | Literal | Sarcastic rate |
| --- | --- | ---: | ---: | ---: | ---: |
| train | `train_sentence.csv` / `train_label.csv` | 39,780 | 18,488 | 21,292 | 0.465 |
| test | `test_sentence.csv` / `test_label.csv` | 2,000 | 1,000 | 1,000 | 0.500 |
| subtest | `subtest_sentence.csv` / `subtest_label.csv` | 278 | 172 | 106 | 0.619 |

The **subtest** is not a random slice of test. It is the emoji-heavy slice
used to show whether emoji2vec helps: 276 / 278 tweets contain at least
one emoji (99.3%). Train and the full test set are only ~14% emoji.

Length (whitespace tokens, train): min 1, median 16, mean 16.5, max 51.
That matches the BiLSTM input length of 78 seen in
`evaluate_loaded_dl_models.ipynb` (padded).

## Distant supervision / hashtag leakage

A large fraction of sarcastic tweets were collected with self-annotation
hashtags. That is standard for Twitter sarcasm datasets, and it is also
a leak if a model can just read the tag.

Cue hashtags used in the examples:

`#not`, `#sarcasm`, `#sarcastic`, `#sarcastictweet`, `#yeahright`,
`#irony`, `#ironic`

On **train**:

* `#not` appears 3,491 times, P(sarcastic | #not) ≈ 0.964
* `#sarcastic` n=287, P ≈ 0.986
* `#sarcastictweet` n=229, P ≈ 0.996
* About 19.7% of sarcastic train tweets contain one of these tags, vs
  0.6% of literal tweets

On **test** the leak is much stronger:

* 64.1% of sarcastic tweets have a cue hashtag vs 0.2% of literal tweets
* 80.4% of sarcastic tweets have *any* hashtag
* Top tags: `#not` (465), `#sarcasm` (89), `#sarcastictweet` (38)

On **subtest**:

* 78.5% of sarcastic tweets have a cue hashtag; 0% of literal tweets do
* Almost every tweet has emoji, so emoji presence alone does not separate
  the classes — emoji *identity* / emoji2vec geometry might

This is why `examples/04_lexicon_baseline.py` exists. A tag lexicon is a
high-precision predictor and looks strong on test/subtest. The 2023
BiLSTM still improves on it, which is the claim worth documenting.

## Other surface stats (train)

| Signal | Overall | Literal | Sarcastic |
| --- | ---: | ---: | ---: |
| Contains emoji | 0.137 | 0.150 | 0.123 |
| Contains a hashtag | 0.213 | 0.167 | 0.266 |
| Contains `<user>` | 0.237 | — | — |

Train still uses the `<user>` placeholder from the original dump. Test and
subtest do not (0 mentions). Top train hashtags after `#not`:
`#yeahright`, `#sarcastictweet`, `#mtvstars`, `#cantwait`,
`#bestfeelingever`.

## Example tweets

Sarcastic (subtest style):

* `I loovee when people text back ... 😒 #sarcastictweet`
* `I just love having grungy ass hair 😑 #not`

Sarcastic without a cue tag (train):

* `Being sore is the best and the worst feeling in the world`

Literal with emoji (train):

* `<user> i hope youre lurking rn. ... pretty please?! 😭 😭 😭`

`#not` is not a perfect feature. Train contains literal lines such as
`100 days until Christmas! 🌲 #too soon #not ready yet` where `#not` is
part of a longer phrase. The lexicon baseline will get those wrong; the
BiLSTM has a chance not to.

## How the 2023 code reads the files

`data_utils.ReadOpen` reads the sentence file as raw lines, replaces
commas with spaces, and tokenizes with NLTK `TweetTokenizer`. Labels
come from pandas `read_csv(..., header=None)`.

`sarcasm_toolkit.dataset.load_split` keeps the original tweet string
(only stripping wrapping quotes) so the inspect example can print text
the way it appears in the CSV.
