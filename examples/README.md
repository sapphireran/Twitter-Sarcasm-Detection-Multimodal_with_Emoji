# Examples

Walkthroughs of the 2023 pipeline that run on **Python 3.10+ and NumPy**. They do not load GloVe, TensorFlow, NLTK, gensim, or the `emoji` package.

Install:

```bash
python3 -m pip install -r requirements-examples.txt
```

Every script accepts `--help`. Scripts that read the real dump take `--root dataset`.

## Library (`examples/lib`)

| Module | What it stands in for |
| --- | --- |
| `dataset_io.py` | `ReadOpen` without pandas |
| `tweet_tokenize.py` | NLTK `TweetTokenizer` subset |
| `emoji_extract.py` | the emoji walk inside `Preprocess` |
| `mean_pool.py` | `AverageVectorPerTweet` / `AverageVectorPerEmoji` / concat |
| `attention_numpy.py` | `attention_layer.Attention.call` |
| `lexical_features.py` | surface cues embeddings can latch onto |
| `logistic.py` | tiny L2 logistic for the cue-only floor |

Import from the repo root (`sys.path` is wired in each CLI):

```python
from examples.lib import tokenize_tweet, attention_forward, load_all_splits
```

## Scripts

### `inspect_dataset.py`

Recomputes the tables in `docs/dataset.md`: sizes, label balance, split overlap, `#not` by class, token-length summaries.

```bash
python3 examples/inspect_dataset.py --root dataset --out docs/generated/dataset_report.md
```

Expect subtest ⊂ test (278/278) and a handful of test∩train collisions.

### `tokenize_tweets.py`

Prints whitespace tokens vs the Twitter-aware splitter on a few sarcastic / sincere lines. Pass `--n 8` to also dump the first eight training rows.

```bash
python3 examples/tokenize_tweets.py --n 5 --from-split test
```

### `emoji_signals.py`

Tweet-level emoji support, P(sarcastic | emoji), and PMI. Co-occurrence pairs are unique-emoji-per-tweet.

```bash
python3 examples/emoji_signals.py --root dataset --out docs/generated/emoji_signals.md
```

Unamused / expressionless faces should show higher P(sarc|e) than seasonal pictographs.

### `embedding_average.py`

Hand-built 8-d “GloVe” and “emoji2vec” tables. Mean-pools five tweets, concatenates to 16-d, and prints cosines. This is `ml_read_data` at toy scale: text-only lines get a **zero emoji half**.

```bash
python3 examples/embedding_average.py
```

### `attention_demo.py`

One batch of `[love, #not, <pad>]` through the Raffel scorer, with and without a pad mask. Exits nonzero if `#not` does not win or if the mask leaks.

```bash
python3 examples/attention_demo.py
```

### `lexical_baseline.py`

Fits L2 logistic regression on the real CSVs using only hand-built cues. Writes accuracy / F1 plus the heaviest weights. This is the “spelling floor” discussed in `docs/results.md`.

```bash
python3 examples/lexical_baseline.py --root dataset --out docs/generated/lexical_baseline.md
```

## Fixtures

`examples/fixtures/sample_{tweets,labels}.csv` is a 10-row miniature split used by unit tests. It is **not** a substitute for `dataset/`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
