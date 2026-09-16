# Examples

Runnable walkthroughs of the coursework ideas. None of these import
TensorFlow, gensim, or nltk. From the repo root:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/inspect_dataset.py
python3 examples/tokenize_tweets.py
python3 examples/toy_embedding_fusion.py
python3 examples/attention_walkthrough.py
python3 examples/lexical_sarcasm_baseline.py
python3 -m unittest discover -s tests -v
```

| Script | What you should see |
| --- | --- |
| `inspect_dataset.py` | 39,780 / 2,000 / 278 rows, class counts, emoji and `#` rates |
| `tokenize_tweets.py` | commas→spaces, then tokens, tags, emoji, elongation |
| `toy_embedding_fusion.py` | 8-d mean word vector ∥ 8-d mean emoji vector |
| `attention_walkthrough.py` | Raffel energies / α peaking on `😒` and `#not` |
| `lexical_sarcasm_baseline.py` | NumPy logreg vs a hashtag-only rule on the real splits |

`lib/` is the shared implementation (`dataset_io`, `tweet_tokenize`,
`embeddings`, `attention_numpy`, `lexical_features`, `logreg`).

The original notebooks still live at the repo root. They need GloVe
and a complete SavedModel; see [../docs/reproduction.md](../docs/reproduction.md).
