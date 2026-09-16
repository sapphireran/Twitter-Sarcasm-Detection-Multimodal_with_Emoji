# Dataset card

The course dump is a three-way split of English tweets labelled for
sarcasm. Files sit in [`dataset/`](../dataset) as parallel
`*_sentence.csv` / `*_label.csv` pairs. There is **no header row**.
Labels are integers: `0` = not sarcastic, `1` = sarcastic.

## Official sizes

Counted from the files in this snapshot (one record per line, labels
aligned 1:1 with sentences):

| Split | Rows | Sarcastic (`1`) | Non-sarcastic (`0`) | Sarcastic rate |
| --- | ---: | ---: | ---: | ---: |
| train | 39,780 | 18,488 | 21,292 | 46.48% |
| test | 2,000 | 1,000 | 1,000 | 50.00% |
| subtest | 278 | 172 | 106 | 61.87% |

`python3 examples/explore_dataset.py` reprints these numbers and adds
token / emoji / hashtag rates.

## How the files are stored

Each sentence file is **one tweet per physical line**, not a
multi-column CSV. A subset of rows is wrapped in double quotes because
the tweet itself contained commas:

```text
"So many useless classes , great to be student"
I just love having grungy ass hair 😑 #not
```

`data_utils.ReadOpen` never ran a CSV parser. It did:

```python
sentence = " ".join(line.strip().split(","))
```

so a quoted comma becomes extra spaces. The example tokenizer prints
both the raw line and that historical transform
(`python3 examples/tokenize_tweets.py`). For analysis that cares about
the original characters, `examples.lib.dataset.load_split` defaults to
leaving the line intact.

## What the splits were for

* **train** — everything the sklearn models and the BiLSTM saw during
  fitting. Slightly more non-sarcastic than sarcastic.
* **test** — balanced official hold-out (2,000 tweets). All headline
  numbers in the course write-up use this split.
* **subtest** — 278 tweets that are denser in emoji and explicit
  sarcasm hashtags. It is *not* a random subsample of test; it is a
  stress test for the emoji channel. The sarcastic prior is higher
  (62%), so accuracy here is not comparable 1:1 with the balanced test.

## Surface cues (training split)

Hashtag counts below come from a full pass over `train_sentence.csv`
(`examples/explore_dataset.py` / `examples/sarcasm_cues.py`).

| Hashtag | Count | P(sarcastic \| tag) |
| --- | ---: | ---: |
| `#not` | 3,191 | 0.974 |
| `#yeahright` | 241 | 0.983 |
| `#sarcastictweet` | 229 | 0.996 |
| `#sarcastic` | 58 | 0.948 |
| `#mtvstars` | 126 | 0.151 |
| `#bestfeelingever` | 78 | 0.013 |

`#not` is almost a gold label when it appears — but it does **not**
appear on most sarcastic tweets. On the balanced test set the
transparent rule in `examples/sarcasm_cues.py` is therefore precise and
low-recall. That is the point of keeping it: the 87% BiLSTM score has
to come from something broader than hashtag lookup.

## Emoji is not a sarcasm detector by itself

On train, the share of tweets that contain at least one emoji-ish
character is slightly *higher* for the non-sarcastic class (~15%) than
for the sarcastic class (~12%). Presence/absence is the wrong feature.
The project instead uses **emoji identity in the same vector space as
words** (emoji2vec aligned to Twitter GloVe) so that `😒` after
"I love this" can sit near other frustrated-face contexts.

The 278-row subtest is where that identity signal is easy to see:
multi-modal forest and BiLSTM both jump several points relative to
word-only, while the balanced test set only moves about one point.

## Mentions and other artefacts

Many rows start with `<user>` rather than `@name`. That is how the
source dump anonymised mentions. `ReadOpen` lower-cases the token, so
the embedding lookup is for `"<user>"`. URLs, leftover HTML, and
hashtag salad (`#as #every #year #am #not #special…`) all appear in
the wild; the tokenizer treats them as ordinary tokens.

## What is *not* in the dump

* No tweet ids, timestamps, or author ids.
* No official train/dev split inside the 39,780 — the notebooks used
  the official test as the validation monitor when evaluating the
  saved Keras models.
* No guarantee that train and test are disjoint at the string level.
  If you need that check, it is a few lines on top of
  `examples.lib.dataset.load_split`.

## License / provenance

The CSVs were uploaded with the course project in 2023
(`d60afa1 Upload dataset and model files`). Treat them as a teaching
dump, not as a claim that the tweets are free of third-party rights.
Do not re-host them as a new "benchmark" without checking the original
collection policy.
