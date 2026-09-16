# Examples

Stdlib-only walkthroughs for the personal CCS2 sarcasm project. None of these
scripts call a social-media API, download GloVe, or import TensorFlow.

Run them from the repository root:

```bash
python3 examples/inspect_dataset.py
python3 examples/tokenize_demo.py --limit 8
python3 examples/emoji_cooccurrence.py --top 15
python3 examples/emoji_cooccurrence.py --strip-leak --min-count 20
python3 examples/heuristic_baseline.py
python3 examples/embedding_average_demo.py
python3 examples/attention_walkthrough.py --prefer leak
python3 examples/attention_walkthrough.py --prefer emoji
```

## What each script is for

| Script | Reads | Shows |
| --- | --- | --- |
| `inspect_dataset.py` | all three CSV splits | sizes, class rates, length, leak-hashtag and emoji counts |
| `tokenize_demo.py` | optional `--text` or `subtest` | hashtag / emoji / `<user>` tokenization |
| `emoji_cooccurrence.py` | one split | sarcastic rate per emoji, with an optional leak-hashtag filter |
| `heuristic_baseline.py` | train + test + subtest | Bernoulli NB on hand features, **with vs without** `#not` |
| `embedding_average_demo.py` | toy table | the same mean / concat rule as `data_utils.py` |
| `attention_walkthrough.py` | synthetic hidden states | Raffel-style alphas on a real tweet’s tokens |

Shared code lives in `examples/lib/`. Tests in `tests/` import that package by
putting `examples/` on `sys.path`.

## Design constraints

- No `pandas`, `numpy`, `sklearn`, `nltk`, `gensim`, or `tensorflow`.
- No network. The only data is `dataset/*.csv`.
- Tokenization is a readable regex approximation of `TweetTokenizer`, not a
  bit-identical port. Use it to understand the pipeline; do not use it to
  claim identical metrics to the 2023 notebooks.
