# Dataset card

This card describes the files already in `dataset/`. It does not introduce a new crawl and it does not download anything from social networks.

## Files

| File | Rows | Role |
| --- | ---: | --- |
| `dataset/train_sentence.csv` | 39,780 | Training tweets, one per line |
| `dataset/train_label.csv` | 39,780 | Training labels (`0` or `1`) |
| `dataset/test_sentence.csv` | 2,000 | Held-out test tweets |
| `dataset/test_label.csv` | 2,000 | Held-out test labels |
| `dataset/subtest_sentence.csv` | 278 | Emoji-rich evaluation slice |
| `dataset/subtest_label.csv` | 278 | Subtest labels |

Row counts match across each sentence / label pair. `examples/inspect_dataset.py` re-computes the tables in this document.

## Label convention

`1` = sarcastic, `0` = not sarcastic.

Evidence from the training file:

- `#not` appears 3,108 times in class 1 and 83 times in class 0
- `#yeahright` (237) and `#sarcastictweet` (228) are almost exclusive to class 1
- The test set repeats the pattern: every one of the 465 `#not` tweets is labeled 1, and `#sarcasm` (89) is class 1 only

Hashtag supervision is a known source of both signal and noise. Some sarcastic-looking tweets without those tags are labeled 0, and a small number of `#not` tweets are labeled 0.

## How the sentence files are stored

The `.csv` suffix is misleading. Each file is **one tweet per line**, not a headered table.

- Some lines are wrapped in ASCII double quotes because the tweet contains a comma.
- Mentions are already anonymized as `<user>`.
- URLs, when present, appear as `<url>` or as a raw `http://` / `https://` string.
- There is no tweet id, timestamp, or user field.

`data_utils.ReadOpen` does not use a CSV parser. It does:

```text
sentence = ' '.join(line.strip().split(','))
```

and then runs NLTK `TweetTokenizer` with `.lower()` on each token. Commas inside a tweet become spaces. The example loader in `examples/sarcasm_lib/dataset.py` can replay that behavior (`faithful=True`) or keep commas and only strip a wrapping quote pair (`faithful=False`, the default for inspection).

## Split statistics

Computed from the files on disk (whitespace-separated word counts, Unicode emoji ranges, `#\w+` hashtags).

### Class balance and length

| Split | n | Class 0 | Class 1 | Mean words | Median words | Max words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 39,780 | 21,292 (53.5%) | 18,488 (46.5%) | 16.46 | 16 | 51 |
| test | 2,000 | 1,000 (50.0%) | 1,000 (50.0%) | 16.51 | 16 | 36 |
| subtest | 278 | 106 (38.1%) | 172 (61.9%) | 17.70 | 17 | 36 |

Training class 0 tweets average 15.88 words; class 1 tweets average 17.13 words. Sarcastic tweets are slightly longer, which is consistent with punchline + tag constructions (`... #not`).

### Surface cues

| Split | Tweets with emoji | Tweets with a hashtag | Tweets with `<user>` | Tweets with `#not` |
| --- | ---: | ---: | ---: | ---: |
| train | 5,470 (13.8%) | 8,486 (21.3%) | 9,433 (23.7%) | 3,188 |
| test | 277 (13.9%) | 932 (46.6%) | 0 | 465 |
| subtest | 277 (99.6%) | 155 (55.8%) | 0 | 110 |

The **subtest is an emoji filter** of the test distribution: 277 of 278 lines contain at least one emoji, and the class prior shifts toward sarcasm (61.9%). That is why the original notebooks report a separate “subtest” number — it is the intended multimodal stress test, not a random 278-row draw.

Training emoji rate is almost the same in both classes (15.0% of class 0, 12.3% of class 1). On test, emoji are *more* common in sarcastic tweets (17.1% vs 10.6%). Subtest removes that comparison by construction.

### Frequent hashtags (train)

Class 0 is dominated by topical or community tags (`#mtvstars`, `#bestfeelingever`, `#whitepeopleproblems`). Class 1 is dominated by sarcasm-marking tags.

| Rank | Class 0 | Count | Class 1 | Count |
| ---: | --- | ---: | --- | ---: |
| 1 | `#mtvstars` | 107 | `#not` | 3,108 |
| 2 | `#not` | 83 | `#yeahright` | 237 |
| 3 | `#bestfeelingever` | 77 | `#sarcastictweet` | 228 |
| 4 | `#whitepeopleproblems` | 71 | `#sarcastic` | 55 |
| 5 | `#cantwait` | 61 | `#cool` | 39 |

`#not` in class 0 is the main reminder that hashtag supervision is imperfect.

### Frequent emoji (train)

| Rank | Class 0 | Count | Class 1 | Count |
| ---: | --- | ---: | ---: | ---: |
| 1 | 😂 | 1,070 | 😂 | 786 |
| 2 | ❤ | 462 | 😊 | 374 |
| 3 | 😍 | 414 | 😒 | 291 |
| 4 | 😭 | 378 | 😭 | 228 |
| 5 | 😊 | 273 | 😍 | 178 |
| 6 | 😩 | 215 | 😩 | 138 |
| 7 | 😘 | 177 | ❤ | 113 |
| 8 | 😒 | 122 | 😑 | 98 |

Face-with-tears-of-joy is the mode in both classes. Unamused (😒) and expressionless (😑) are more characteristic of class 1; hearts and smiling-face-with-heart-eyes lean class 0. The test set exaggerates this: 😒 is the top sarcastic emoji (33) while 😂 remains the top non-sarcastic emoji (25). The gun emoji (🔫) appears 13 times in sarcastic test tweets and is a good example of why emoji *sense* (deadpan threat, mock despair) is not the same as emoji *presence*.

`examples/emoji_signals.py` scores tokens with a smoothed log-odds ratio so the ranking is not just raw frequency.

## Example lines

Non-sarcastic (label 0), training file:

```text
<user> i hope youre lurking rn. i want to listen to hallucination & wanna love you again live someday, pretty please?! 😭 😭 😭
<user> Rest in peace & love to you and your family
```

Sarcastic (label 1), training file:

```text
Being sore is the best and the worst feeling in the world
Expecting is my favorite crime and disappointment is always my punishment.. 😎😎 😥 😊
Happy birthday to me. Yay.
```

These are illustrative, not a claim that every short congratulation is sarcastic. The label is whatever the file stores.

## Intended use

- Binary sarcasm classification research and course-project writeups
- Comparing text-only vs text+emoji representations on the same split
- Documenting preprocessing quirks in the original 2023 code

Not intended for:

- Retrieving the original accounts or reconstructing deleted posts
- Training a production moderation system
- Claiming a new state-of-the-art without a documented external benchmark

## Provenance

The README in the first commit describes the work as a UCPH Computational Cognitive Science 2 (2023) final project. The tweets look like a hashtag-supervised Twitter sarcasm collection (the `#not` / `#sarcasm` / `#yeahright` pattern is standard in that literature). This card does not assign a specific public corpus name because the repo does not record one.

## Related local artifacts

- `emoji2vec.bin` — generic emoji2vec binary
- `emoji2vec_twitter.bin` — Twitter-tuned emoji2vec binary used by the notebooks
- GloVe Twitter 27B 200-d — **not in the repo**; required only by the original notebooks
