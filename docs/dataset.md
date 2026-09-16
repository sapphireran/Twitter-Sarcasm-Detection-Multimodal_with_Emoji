# Dataset

All tweets live as **one sentence per line** next to a parallel **one integer label per line** file (`0` = not sarcastic, `1` = sarcastic). There is no header row.

| File | Lines | Role |
| --- | ---: | --- |
| `dataset/train_sentence.csv` / `train_label.csv` | 39,780 | Fit tokenizers, embeddings, and classifiers |
| `dataset/test_sentence.csv` / `test_label.csv` | 2,000 | Held-out report split (balanced) |
| `dataset/subtest_sentence.csv` / `subtest_label.csv` | 278 | Emoji-bearing slice of **test** |

The `.csv` suffix is historical. `data_utils.ReadOpen` does **not** parse commas as columns. It reads raw lines, then *joins on spaces after splitting on commas*, which is a light way to flatten `"quoted, tweet, text"` rows. Labels are loaded with `pandas.read_csv(..., header=None)`.

## Label balance

| Split | `#0` | `#1` | Positive rate |
| --- | ---: | ---: | ---: |
| Train | 21,292 | 18,488 | 46.5% |
| Test | 1,000 | 1,000 | 50.0% |
| Subtest | 106 | 172 | 61.9% |

Train is slightly skewed toward the negative class. Test was balanced on purpose. Subtest is not balanced: emoji-bearing sarcastic tweets are over-represented relative to emoji-bearing non-sarcastic ones.

## How the splits relate

Computed from the checked-in CSVs (see `examples/inspect_dataset.py`):

- **Subtest ⊂ test.** All 278 subtest strings appear in `test_sentence.csv`. Subtest is the subset of test tweets that contain at least one codepoint above U+2710 (a cheap “looks like emoji / dingbat / extra symbol” filter). That is why notebooks treat subtest as the *emoji-conditioned* evaluation, not as a third independent draw.
- **Train ≉ test.** 48 test strings also appear in train (22 unique train tweets are duplicated inside train itself). That is a small leak (~2.4% of test). It does not change the ranking of models in the 2023 tables, but it does mean test accuracy is a slightly optimistic estimate.
- Mentions in train are anonymized as the token `<user>` (9,433 training lines). Test and subtest have **zero** `<user>` tokens — a domain shift the frozen GloVe row for `<user>` cannot help at eval time.

## Surface statistics (train)

| Cue | All train | Sarcastic | Non-sarcastic |
| --- | ---: | ---: | ---: |
| Contains `#` | 8,502 | 4,929 | 3,573 |
| Contains `#not` (word boundary) | 3,188 | **3,105** | 83 |
| Contains `#sarcas*` | 287 | (almost all positive) | — |
| Contains `<user>` | 9,433 | — | — |
| High-codepoint characters | 5,360 | — | — |
| Mean whitespace tokens | 16.5 | — | — |
| Max whitespace tokens | 51 | — | — |

`#not` is the strongest single lexical fire alarm in this dump. A model that only looks at embeddings can still latch onto the GloVe neighborhood of `not` / `#not` rather than “understanding” irony. The lexical baseline in `examples/lexical_baseline.py` makes that shortcut explicit so the neural results have a floor to beat.

Whitespace tokenization under-counts emoji clusters and hashtags glued to punctuation. `examples/tokenize_tweets.py` uses a Twitter-aware splitter closer to NLTK `TweetTokenizer`, which is what `ReadOpen` actually calls.

## What a row looks like

Anonymized / shortened examples from the checked-in files:

| Label | Split | Text (truncated) |
| ---: | --- | --- |
| 1 | test | `I loovee when people text back ... 😒 #sarcastictweet` |
| 1 | test | `Oh how I love getting home from work at 3am and my house being dirty #not` |
| 1 | train | `100% failed both physics exams great` |
| 0 | train | `100 days until Christmas! 🌲 #too soon #not ready yet` |
| 0 | test | `Want to have someone to speak to I'm so bored 😭` |

Notice the last non-sarcastic train example contains the **substring** `not` inside `#not ready` — the hashtag tokenizer in `ReadOpen` will usually emit `#not` as its own token, which is why cue counts must use a real tweet tokenizer, not raw `in` checks, when you care about precision.

## Embedding coverage (original run)

The metrics notebook printed:

```
Loaded 1193515 word vectors.
0 words not found in vocabulary
```

That “0 not found” line is about the **GloVe file load**, not about every tweet token hitting the table. `Preprocess` still writes a zero row (or an emoji2vec mean) for OOV tokens. Emoji fallback is the whole multi-modal story: `emoji.emoji_list(word)` + `emoji.is_emoji(c)` walks a token, gathers emoji2vec rows, and averages them into the same 200-d slot GloVe would have filled.

`emoji2vec.bin` and `emoji2vec_twitter.bin` are checked in (~2.0 MB and ~1.3 MB). The Twitter-tuned file is what the notebooks load.

## Provenance

This dump was assembled for the 2023 CCS2 project (self-collected / course-provided Twitter sarcasm labels; mentions stripped). It is **personal course data**, not a company corpus. Do not treat it as a general-purpose Twitter firehose: it is short, English-leaning, sarcasm-hashtag-heavy, and already anonymized.

## Regenerating the census

```bash
python3 examples/inspect_dataset.py --root dataset --out docs/generated/dataset_report.md
```

The script re-computes every table in this page from the CSVs so the markdown cannot drift from the files.
