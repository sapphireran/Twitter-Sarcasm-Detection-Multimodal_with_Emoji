# Dataset

The corpus is three aligned pairs of CSVs under [`dataset/`](../dataset/):

| File | Role |
| --- | --- |
| `train_sentence.csv` / `train_label.csv` | Fit vocabularies, embedding rows, and classifiers |
| `test_sentence.csv` / `test_label.csv` | Balanced held-out set used for the main table |
| `subtest_sentence.csv` / `subtest_label.csv` | Small emoji-heavy slice used as a stress test |

There is no official datasheet in the 2023 upload. Everything below is measured from the files in this clone.

## File format

- **Sentences:** one tweet per line, UTF-8. Some lines are wrapped in ASCII double quotes because the original export treated commas as CSV separators. `ReadOpen` in `data_utils.py` joins on spaces after `split(',')`, which is a lossy way to undo that quoting. The lightweight example tokenizer keeps the comma instead; both behaviors are demonstrated in `examples/02_tokenize_tweets.py`.
- **User mentions:** many training tweets contain the literal token `<user>` rather than a raw `@handle`.
- **Labels:** a single integer per line, `0` = not sarcastic, `1` = sarcastic. No header row.
- **Alignment:** line *i* of `*_sentence.csv` belongs with line *i* of `*_label.csv`. `examples/07_split_consistency.py` checks that pairing.

## Split sizes

| Split | Lines | Sarcastic | Non-sarcastic | Sarcastic % | Median tokens (whitespace) | Min / max tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 39,780 | 18,488 | 21,292 | 46.5 | 16 | 1 / 51 |
| test | 2,000 | 1,000 | 1,000 | 50.0 | 16 | 1 / 36 |
| subtest | 278 | 172 | 106 | 61.9 | 17 | 5 / 36 |

Train is slightly majority-negative. Test was balanced. Subtest is majority-sarcastic, which is why F1 and accuracy can move together there even when a model is biased toward the positive class.

## Why the subtest exists

The project title is multimodal **with emoji**. Most tweets in train/test do not contain an emoji at all:

| Split | Tweets with at least one emoji-range codepoint | Among sarcastic | Among non-sarcastic |
| --- | ---: | ---: | ---: |
| train | 5,223 (13.1%) | 2,198 | 3,025 |
| test | 266 (13.3%) | 164 | 102 |
| subtest | 266 (95.7%) | 164 | 102 |

The emoji-positive counts on test and subtest match exactly (164 sarcastic, 102 not). Subtest is 278 rows, so it is the emoji-bearing test tweets plus a handful of near-neighbors. Evaluating only on `test` therefore dilutes the modality the project set out to study; evaluating only on `subtest` over-represents sarcastic emoji tweets. The original notebooks report both.

## Explicit sarcasm cues

Hashtags that *name* the label are common. Counting `#not`, `#sarcasm`, `#sarcastic`, `#sarcastictweet`, and `#irony` / `#ironic` (case-insensitive):

| Split | Tweets with a cue hashtag | Cue + sarcastic | Cue + non-sarcastic |
| --- | ---: | ---: | ---: |
| train | 3,480 | 3,392 | 88 |
| test | 595 | 595 | 0 |
| subtest | 129 | 129 | 0 |

On test, every cue-hashtag tweet is labeled sarcastic. That is a leak if a model can see raw hashtag strings, and a useful baseline feature if you are honest about it. The 2023 neural model sees hashtags as tokens in GloVe space (`#not` may or may not be in-vocab). The toy cue classifier in `examples/06_toy_baseline.py` uses the leak on purpose so the examples can show a ceiling that does not require embeddings.

Most frequent hashtags, train split:

| Hashtag | Count |
| --- | ---: |
| `#not` | 3,191 |
| `#yeahright` | 241 |
| `#sarcastictweet` | 229 |
| `#mtvstars` | 126 |
| `#cantwait` | 94 |
| `#bestfeelingever` | 78 |
| `#whitepeopleproblems` | 71 |
| `#fail` | 59 |
| `#sarcastic` | 58 |

Most frequent hashtags, test split: `#not` (465), `#sarcasm` (89), `#sarcastictweet` (38), `#yeahright` (19).

## Mentions and surface form

- **Train** has 9,433 tweets containing `<user>` or `@…`.
- **Test** has almost no mention markup (5 tweets). The test distribution is therefore not a random slice of train on that axis.
- Tweets are short. A 16-token median is why a two-layer BiLSTM is not as extravagant as it looks, and why mean-pooling (the sklearn path) throws away order that the recurrent path can use.

## Worked examples from `test_sentence.csv`

Sarcastic, emoji, cue hashtag:

```text
I loovee when people text back ... 😒 #sarcastictweet          → 1
I just love having grungy ass hair 😑 #not                     → 1
feeling like a million bucks after that chem 2 test . 😅 #not  → 1
```

Sarcastic without a cue hashtag (harder; needs polarity reversal in the words themselves):

```text
So many useless classes , great to be student                  → 1
Thank you , random guy , for sneaking up behind me ,
and grabbing my face . That was no invasion of privacy
whatsoever .                                                   → 1
```

Non-sarcastic with emoji (emoji is affect, not a sarcasm marker):

```text
Want to have someone to speak to I'm so bored 😭               → 0
I feel so honored ... want me to come speak at their school 😌 → 0
```

## What this dataset is not

- It is not a streaming Twitter / X API dump. Handles are already stripped or replaced.
- It is not parallel image data. “Multimodal” here means **word embedding + emoji embedding**, not text+photo.
- It is not balanced on emoji presence. Any claim of the form “emoji2vec always helps” has to be qualified by split.

`examples/01_dataset_overview.py` reprints the tables on this page from the CSVs so the numbers stay auditable.
