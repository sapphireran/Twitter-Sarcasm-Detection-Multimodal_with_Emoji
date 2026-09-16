# Personal examples

These scripts replay the 2023 project’s *ideas* without GloVe, TensorFlow, or
NLTK. They are meant to be read top to bottom, then run.

```bash
python3 -m pip install -r requirements-examples.txt   # NumPy
python3 examples/01_dataset_overview.py
python3 examples/02_tokenize_tweets.py
python3 examples/03_sarcasm_cues.py
python3 examples/04_tiny_embedding_pipeline.py
python3 examples/05_attention_math.py
python3 examples/06_toy_baseline.py
python3 examples/07_split_consistency.py
python3 examples/08_format_results_table.py
```

`lite_pipeline.py` is the shared library. It is a stand-in, not a drop-in
replacement, for `data_utils.py`. Differences are listed in
[`docs/preprocessing.md`](../docs/preprocessing.md).

## What each script is for

| Script | Reads | Shows |
| --- | --- | --- |
| `01_dataset_overview.py` | `dataset/*.csv` | Split sizes, emoji rate, cue-hashtag rate, top hashtags |
| `02_tokenize_tweets.py` | fixture + one test line | Keep-commas vs 2023 comma-smash tokenization |
| `03_sarcasm_cues.py` | `dataset/*.csv` | Precision / recall of “`#not` ⇒ sarcastic” |
| `04_tiny_embedding_pipeline.py` | `fixtures/tiny_*.csv` | 16-d mean-pool word view vs 32-d concat view |
| `05_attention_math.py` | nothing | Raffel attention identities (mask, Σa=1, h=Σax) |
| `06_toy_baseline.py` | `dataset/*.csv` | Nearest-centroid on 4 hand features; cue rule |
| `07_split_consistency.py` | `dataset/*.csv` | Alignment + the counts frozen in the docs |
| `08_format_results_table.py` | `docs/results/recorded_metrics.json` | Markdown tables of the 2023 notebook numbers |

## Fixture

`fixtures/tiny_sentence.csv` / `tiny_label.csv` are twelve lines taken from
or paraphrased after the public CSVs in this repo: six sarcastic, six not,
mix of cue hashtags, emoji, `<user>`, and comma-bearing text. Use them when
you do not want to stream 39k train tweets.

## What these examples will not do

- They will not load `model/best_model_*` or the sklearn pickles.
- They will not download GloVe-Twitter.
- They will not post anything or call a social API. The corpus is already
  on disk as CSV.

The original lab path is documented in [`docs/reproducing.md`](../docs/reproducing.md).
