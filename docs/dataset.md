# Dataset

All splits live under `dataset/`. Each split is a pair of
line-aligned files:

| File | Rows | Role |
| --- | ---: | --- |
| `train_sentence.csv` / `train_label.csv` | 39,780 | Fit tokenizers, embeddings, and classifiers |
| `test_sentence.csv` / `test_label.csv` | 2,000 | Held-out full test (balanced) |
| `subtest_sentence.csv` / `subtest_label.csv` | 278 | Emoji-heavy diagnostic slice |

Labels are a single integer per line:

- `0` — not sarcastic
- `1` — sarcastic

Sentence files are not a strict CSV table. Many lines are raw tweets;
some are quoted because they contain commas. `ReadOpen` in
`data_utils.py` strips the line, splits on commas, then joins the
pieces back with spaces before tokenization. That is a blunt way to
undo CSV quoting rather than a real CSV parser.

## Class balance

Computed from the checked-in files (see
`examples/inspect_dataset.py`):

| Split | n | Not sarcastic (0) | Sarcastic (1) | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| train | 39,780 | 21,292 | 18,488 | 46.5% |
| test | 2,000 | 1,000 | 1,000 | 50.0% |
| subtest | 278 | 106 | 172 | 61.9% |

Train is slightly majority-negative. Test was constructed to be
balanced, so accuracy and F1 are easy to compare. Subtest is
majority-sarcastic and should not be treated as a second i.i.d. test
set — it is a **stress test for emoji and sarcasm-hashtag cues**.

## Length

Whitespace token counts (before NLTK `TweetTokenizer`):

| Split | Mean | Median | Max |
| --- | ---: | ---: | ---: |
| train | 16.5 | 16 | 51 |
| test | 16.5 | 16 | 36 |
| subtest | 17.7 | 17 | 36 |

The Keras path pads to the **training** maximum sequence length after
`TweetTokenizer` + `Tokenizer.texts_to_sequences`. The saved model
summaries show length **78**, which is longer than raw whitespace
counts because emoji, punctuation, and user mentions become their own
tokens.

## Surface cues

| Split | Has `#…` | Has `<user>` | Has `#not` / `#sarcasm` / `#sarcastic(tweet)` | Has non-ASCII |
| --- | ---: | ---: | ---: | ---: |
| train | 8,486 | 9,433 | 3,475 | 5,794 |
| test | 932 | 0 | 594 | 286 |
| subtest | 155 | 0 | 129 | **278** |

Notes that matter for modeling:

1. **User mentions were already anonymized** to the literal token
   `<user>` in train. The test and subtest files in this checkout
   contain no `<user>` tokens.
2. **Every subtest tweet is non-ASCII.** That is why it is the
   emoji-channel diagnostic set.
3. Explicit sarcasm hashtags are almost perfectly aligned with the
   positive class on test/subtest (594/594 and 129/129). On train the
   match is strong but not perfect (3,388 of 3,475 tagged tweets are
   labeled sarcastic; 87 tagged tweets are labeled 0).

A classifier that only looks at `#not` / `#sarcasm` will look
artificially strong on test and especially on subtest. That is why
the project also reports a full-test number, and why the deep model
is still the interesting result: it beats the hashtag-only cue on the
balanced test set.

## Label vs. sarcasm hashtag

| Split | Tag ∧ label=1 | Tag ∧ label=0 |
| --- | ---: | ---: |
| train | 3,388 | 87 |
| test | 594 | 0 |
| subtest | 129 | 0 |

On test, 594 / 1,000 sarcastic tweets (59.4%) carry an explicit tag.
The other 40.6% have to be recovered from wording and emoji alone.

## Emoji are not a sarcasm switch

`examples/emoji_signal.py` (walkthrough tokenizer, not NLTK) measures
raw association. The useful slice is **emoji present, sarcasm hashtag
absent** — that is where emoji2vec has to do work a tag cannot.

| Split | Base P(sarc) | P(sarc \| emoji) | P(sarc \| emoji, no tag) | n(emoji, no tag) |
| --- | ---: | ---: | ---: | ---: |
| train | 0.465 | 0.411 | 0.336 | 5,051 |
| test | 0.500 | 0.606 | 0.277 | 155 |
| subtest | 0.619 | 0.619 | 0.289 | 149 |

On train, having any emoji is slightly *anti*-correlated with the
positive label (phi ≈ −0.044). On test/subtest, emoji-only tweets
are still well below the base rate. Identity matters more than
presence:

| Token (test) | Tweets | P(sarc \| token) |
| --- | ---: | ---: |
| 😒 | 33 | 0.879 |
| 😅 | 16 | 0.875 |
| 🔫 | 11 | 1.000 |
| 😂 | 27 | 0.407 |
| 😘 (train) | 195 | 0.174 |
| ❤ (train) | 428 | 0.220 |

That is the point of a 200-d emoji vector instead of a binary
`has_emoji` feature: 😒 after “I love …” is a different signal from
😘 after a sincere compliment. The walkthrough tokenizer also emits
a lone variation selector `️` for some composed emoji; NLTK
`TweetTokenizer` usually keeps those attached. Treat the top-token
table as directional, not as a gold emoji vocabulary.

## Example tweets

These are real lines from the checked-in files, truncated for
readability.

**Test, sarcastic (1)**

- `I loovee when people text back ... 😒 #sarcastictweet`
- `Don't you love it when your parents are Pissed because you were gonna study after bubble soccer ! #IKnowIDo #not 😃 🔫`
- `Oh how I love getting home from work at 3am and my house being dirty #not`
- `I just love having grungy ass hair 😑 #not`

**Train, not sarcastic (0)**

- `<user> i hope youre lurking rn. i want to listen to hallucination & wanna love you again live someday, pretty please?! 😭 😭 😭`
- `100 days until Christmas! 🌲 #too soon #not ready yet`
- `<user> Rest in peace & love to you and your family`

The Christmas example is a reminder that `#not` is not a clean
boolean feature: "not ready yet" is compositional English, not a
sarcasm tag. `ReadOpen` lowercases after `TweetTokenizer`, so
`#Not` and `#not` collapse, but `#not ready` still has to be
disambiguated by context.

**Train, sarcastic (1)**

- `Being sore is the best and the worst feeling in the world`
- `Expecting is my favorite crime and disappointment is always my punishment.. 😎😎 😥 😊`
- `Happy birthday to me. Yay.`

Those last three have no `#sarcasm` tag. They are the cases where
GloVe polarity plus (sometimes) emoji vectors have to do the work.

## How the original loader treats a line

`data_utils.ReadOpen(filename, Labelfile)`:

1. Read the sentence file as UTF-8 with `errors="replace"`.
2. For each line: `sentence = ' '.join(line.strip().split(','))`.
3. `TweetTokenizer().tokenize(sentence)`, then `.lower()` each token.
4. Read labels with `pandas.read_csv(..., header=None)` and squeeze.

`ml_read_data` then averages GloVe rows over in-vocabulary tokens
(200-d) and, separately, averages emoji2vec rows over in-vocabulary
emoji tokens (200-d). The WE matrix is the concatenation (400-d).
Tweets with no in-vocabulary tokens become a zero vector of the same
width. A single random permutation is applied to both views so W and
WE stay aligned.

The deep path (`Preprocess`) does **not** average. It builds a Keras
word index, pads on the right, and writes a `(vocab, 200)` embedding
matrix. OOV words are scanned with `emoji.emoji_list`; any extracted
emoji are averaged from emoji2vec when `get_emoji2vec=True`.

Run `python3 examples/preprocess_walkthrough.py` to see those steps on
a handful of tweets without loading GloVe.
