# Dataset

All rows live under [`dataset/`](../dataset/). Sentences and labels are
**parallel files** (same line index), not a single two-column CSV.

| File | Rows | Label `0` (literal) | Label `1` (sarcastic) |
| --- | ---: | ---: | ---: |
| `train_sentence.csv` / `train_label.csv` | 39,780 | 21,292 (53.5%) | 18,488 (46.5%) |
| `test_sentence.csv` / `test_label.csv` | 2,000 | 1,000 | 1,000 |
| `subtest_sentence.csv` / `subtest_label.csv` | 278 | 106 (38.1%) | 172 (61.9%) |

The subtest is almost entirely emoji-bearing (277 / 278). It is the
split used in the write-up to argue that emoji2vec helps when emoji
are actually present.

## File format

- Encoding: UTF-8, with `errors="replace"` in `ReadOpen`.
- No header on either file.
- One tweet per line in `*_sentence.csv`. Some lines are wrapped in
  ASCII double quotes because the original export treated commas as
  CSV separators.
- One integer per line in `*_label.csv`: `0` or `1`.
- Mentions are already rewritten as the token `<user>`.
- Typical length: ~16 whitespace tokens, max 51 on train, max 36 on
  test / subtest. Character length on test is roughly 3–156
  (mean ≈ 79).

`data_utils.ReadOpen` does something easy to miss:

```python
sentence = ' '.join(line.strip().split(','))
```

Commas become spaces **before** `nltk.TweetTokenizer` runs. A tweet
like `"So many useless classes , great to be student"` is therefore
tokenized as if the comma were already gone. The example tokenizer
in `examples/lib/tweet_tokenize.py` keeps the same first step so
counts stay comparable.

## Surface cues (test split)

Computed by `examples/inspect_dataset.py`:

| Cue on the 2,000-row test set | Count |
| --- | ---: |
| Contains `#` | 935 |
| `#not` | 465 |
| `#sarcasm` / `#sarcastictweet` / `#sarcastic` | 130 |
| `#yeahright` | 19 |
| Any emoji (common Unicode blocks) | 277 |
| `<user>` or `@` | 5 |

Train is less hashtag-saturated (8,502 / 39,780 rows have `#`) and
more mention-heavy (9,500 rows contain `<user>` or `@`). That
mismatch matters: a hashtag-only rule looks stronger on test than it
would on a mention-heavy timeline.

`examples/inspect_dataset.py` also scores the rule
`#not` / `#sarcasm*` / `#yeahright` → sarcastic. On the **test**
split that rule has precision 1.000 and recall 0.613 (accuracy
0.806). On **train** precision is still 0.976 but recall drops to
0.196 — only 3,191 train tweets carry `#not`, versus 465 of 2,000
test tweets. Models fit on train therefore see a weaker tag signal
than the test set rewards.

The most common test hashtags, in order: `#not`, `#sarcasm`,
`#sarcastictweet`, `#yeahright`, `#fml`, `#exhausted`, `#yay`.

## How the original loader builds vectors

`ml_read_data` (classical models):

1. Tokenize every tweet with `TweetTokenizer`, lowercase.
2. Mean-pool GloVe rows for tokens in `glove_model.vocab` → 200-d.
   Empty tweets become a zero vector.
3. Mean-pool emoji2vec rows the same way → 200-d.
4. Concatenate → 400-d multi-modal vector.
5. Shuffle train/test/subtest independently with
   `numpy.random.permutation` (no seed).

`Preprocess` (deep model):

1. Fit a Keras `Tokenizer` on the **training** token lists.
2. `pad_sequences(..., padding='post')`; `maxlen` is the longest
   train tweet after that (78 in the saved notebooks).
3. Build a `(vocab_size, 200)` matrix. Prefer GloVe; if the token
   looks like emoji, average the emoji2vec rows instead; otherwise
   zeros.
4. Test / subtest sequences are encoded with the **train** tokenizer
   and padded to the **train** `maxlen`.

The embedding-matrix path is “multi-modal in one table”: emoji and
words share the same 200-d space that the LSTM reads. The classical
path is “two mean vectors glued together.”

## Example rows

Sarcastic, emoji + explicit tag (test):

```
I loovee when people text back ... 😒 #sarcastictweet
```

Sarcastic, `#not` + positive polarity (test):

```
Oh how I love getting home from work at 3am and my house being dirty #not
```

Non-sarcastic, emoji, no sarcasm hashtag (test):

```
Want to have someone to speak to I'm so bored 😭
```

More annotated rows: [annotated_examples.md](annotated_examples.md).

## What this dataset is not

- It is not a live Twitter API dump. Do not treat the CSV as
  something you can refresh with the X/Twitter tools.
- It is not balanced the same way across splits (train slightly
  literal-heavy, subtest sarcastic-heavy).
- It is not free of label-hashtag leakage. Many sarcastic test tweets
  announce the label with `#not` / `#sarcasm`.
