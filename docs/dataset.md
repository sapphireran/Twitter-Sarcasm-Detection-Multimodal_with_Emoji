# Dataset notes

This page describes the CSV splits that ship in `dataset/`. Counts were recomputed from the files in this checkout. They are not taken from memory.

## Files

Each split is a pair of line-aligned files:

| Split | Sentence file | Label file | Rows |
| --- | --- | --- | ---: |
| Train | `dataset/train_sentence.csv` | `dataset/train_label.csv` | 39,780 |
| Test | `dataset/test_sentence.csv` | `dataset/test_label.csv` | 2,000 |
| Subtest | `dataset/subtest_sentence.csv` | `dataset/subtest_label.csv` | 278 |

A sentence file is **not** a well-formed RFC 4180 table. Each line is one tweet. Some lines are wrapped in quotes because the original tweet contained commas; `data_utils.ReadOpen` joins comma-separated pieces back into a single string before tokenizing. Labels are a single column of `0` / `1` with no header.

**Convention used in this project:** `1` is sarcastic, `0` is not.

## Class balance

| Split | Sarcastic | Not sarcastic | Positive rate |
| --- | ---: | ---: | ---: |
| Train | 18,488 | 21,292 | 0.465 |
| Test | 1,000 | 1,000 | 0.500 |
| Subtest | 172 | 106 | 0.619 |

The official test set is balanced. The subtest is not: it was assembled as a **harder, emoji-heavy slice**, not as a second i.i.d. draw from the same distribution. Comparing test vs subtest therefore mixes “does emoji help?” with “did we pick the emoji-rich sarcastic tweets?”. That is useful for a course analysis and dangerous if you treat subtest as a leaderboard.

## Length

Approximate whitespace token counts on the raw lines (before `TweetTokenizer`):

| Split | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| Train | 1 | 16 | 16.46 | 51 |
| Test | 1 | 16 | 16.51 | 36 |
| Subtest | 5 | 17 | 17.70 | 36 |

Tweets are short. A 256-unit BiLSTM is oversized relative to sequence length; the capacity is there to mix bidirectional context and then let attention pick a few tokens, not to model long documents.

## Anonymization and surface cues

User mentions in the dump are already replaced with the token `<user>`. Hashtags, emoji, and URLs are left intact. That means a model can (and, in the classical baselines, will) latch onto:

- explicit sarcasm hashtags
- emoji codepoints
- residual `@` / `<user>` patterns
- very short all-caps interjections

It cannot recover the original author graph. There is no social network feature in this repo.

## Hashtags

Hashtag counts on the **training** sentences (case-folded):

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
| `#challengeaccepted` | 57 |

A linear scan for `#not`, `#sarcasm`, or `#sarcastic` hits **3,246** training tweets. That is about 8% of the split and a much larger share of the sarcastic class. Any model that sees raw hashtags is partly solving “detect the sarcasm hashtag,” which is a different task from “detect sarcasm in the wild after those hashtags are stripped.”

The examples script `examples/heuristic_baseline.py` reports accuracy **with and without** those tags so the leakage is visible rather than hidden.

## Emoji

A Unicode-range scan of the training file finds **5,458** tweets with at least one emoji-like codepoint (~13.7%). The most frequent symbols:

| Emoji | Approx. count |
| --- | ---: |
| 😂 | 1,868 |
| 😊 | 668 |
| 😭 | 627 |
| 😍 | 626 |
| ❤ | 612 |
| 😒 | 417 |
| 😩 | 371 |
| 😘 | 243 |
| 😁 | 189 |
| ☺ | 185 |
| 😑 | 169 |
| 😅 | 168 |

Frequency is not polarity. 😂 appears in both sincere and sarcastic tweets. 😒 and 😑 are more concentrated on sarcastic lines in this dump; 😍 and ❤ are more mixed. `examples/emoji_cooccurrence.py` prints polarity rates per emoji so you can see the association without fitting a network.

## What the three splits are for

- **Train** — fit sklearn models and the Keras embedding matrix / tokenizer.
- **Test** — the balanced hold-out used for the headline table.
- **Subtest** — a small, emoji-dense probe. In the original notebooks it is the split where multimodal random forest and BiLSTM pull ahead most clearly.

The notebooks shuffle training rows inside `ml_read_data` (`np.random.permutation`). They do **not** reshuffle labels independently of features; the same index permutation is applied to `X` and `X_emoji`.

## Provenance

The files were uploaded to this personal GitHub repo in 2023 as part of the CCS2 final project. The linguistic genre is the standard Twitter sarcasm setting used in that course (short posts, `#not` / `#sarcasm` style distant supervision, `<user>` anonymization). This documentation does not re-host or re-collect any tweets. Do not use the CSVs to contact users or to reconstruct a live timeline.

## Practical issues when you load the files

1. `ReadOpen` opens with `encoding="utf-8", errors="replace"`. Broken bytes become the replacement character instead of crashing.
2. `pandas.read_csv(..., header=None)` is used only for labels. Sentence files are read with `readlines()` so a quoted comma does not create extra columns.
3. Several notebooks pass `"train_sentence.csv"` (repo root) instead of `"dataset/train_sentence.csv"`. Copy or symlink the CSVs if you re-run those cells unchanged. The examples scripts always use `dataset/`.
4. The first 20 training labels are all `0`. That is a file-order artifact, not a bug in the reader. Always look at the global counts, not the head of the file.
