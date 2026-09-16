# Examples

Small, NumPy-only walkthroughs of the 2023 sarcasm-detection pipeline.
They do not load GloVe, TensorFlow, NLTK, or the SavedModels.

Run every script from the **repository root** so `dataset/` and
`examples/fixtures/` resolve:

```bash
python examples/dataset_overview.py
python examples/tokenize_demo.py
python examples/embedding_demo.py
python examples/attention_demo.py
python examples/sarcasm_cues.py
python examples/toy_pipeline.py
```

Or one shot:

```bash
python examples/run_all.py
python -m unittest discover -s tests -v
```

## What each script is for

| Script | Mirrors | What you should see |
| --- | --- | --- |
| `dataset_overview.py` | row counts the notebooks assume | 39780 / 2000 / 278 splits, emoji rates, `#not` counts |
| `tokenize_demo.py` | `ReadOpen` + `TweetTokenizer` | hashtags / emoji kept; quoted commas broken the 2023 way |
| `embedding_demo.py` | `AverageVectorPerTweet` / `AverageVectorPerEmoji` | 8-d word mean, 8-d emoji mean, 16-d concat |
| `attention_demo.py` | `attention_layer.Attention` | weights on `#not` vs a uniform mean-pool |
| `sarcasm_cues.py` | why subtest ≠ test | emoji and sarcasm-hashtag crosstabs by label |
| `toy_pipeline.py` | `ml_read_data` + a linear classifier | single-modal vs multi-modal on 16 fixture tweets |

`--json` is available on the two analysis scripts if you want to pipe
numbers into another tool.

## Library modules

| Module | Responsibility |
| --- | --- |
| `tokenize.py` | Regex tweet tokenizer + the comma-rebuild quirk |
| `emoji.py` | Pictograph detector used by the analysis scripts |
| `dataset_io.py` | CSV readers for `dataset/` and the fixtures |
| `embeddings.py` | Mean-pool, concat, embedding-matrix fallback |
| `attention.py` | Raffel scores / masked softmax / context vector |
| `classify.py` | NumPy logistic regression, nearest centroid, acc / F1 |
| `fixtures/tiny_tables.py` | Named 8-d word and emoji axes |
| `fixtures/tiny_tweets.csv` | 16 tweets taken from the real corpus style |

## Fixture axes

The toy vectors are not GloVe. Each dimension is a named cue so the
printed weights in `toy_pipeline.py` stay readable:

| Index | Word table | Emoji table |
| ---: | --- | --- |
| 0 | positive surface words (`love`, `happy`) | unused |
| 1 | negative surface words (`hate`, `annoyed`) | unused |
| 2 | sarcasm hashtags (`#not`, `#sarcastictweet`) | unused |
| 3 | school / work grind (`test`, `monday`) | unused |
| 4 | genuine affection (`peace`, `family`) | 🌲 |
| 5 | unused | 😭 😅 😔 |
| 6 | unused | 😒 😑 |
| 7 | unused | 😄 😃 |

A 16-d multi-modal row is `[word 0–7 | emoji 0–7]`.

## Tests

`tests/test_examples.py` checks tokenizer behavior, the known split
sizes, attention masking, embedding OOV zeros, and that the toy
pipeline produces 0/1 predictions. It reads the real `dataset/` CSVs
for the split-size assertions.
