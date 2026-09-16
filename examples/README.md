# Examples

Numbered scripts that run with **Python 3.12 + numpy**. They do not import
TensorFlow, Gensim, NLTK, or pandas. Shared helpers live in
`examples/common/`.

| Script | What it shows |
| --- | --- |
| `01_explore_dataset.py` | Split sizes, class balance, emoji / hashtag / cue rates |
| `02_emoji_cooccurrence.py` | Document-frequency PMI of pictographs vs. the sarcastic class |
| `03_lexical_cues.py` | How far a `#sarcasm` / `#not` rule gets you |
| `04_inspect_emoji2vec.py` | Nearest neighbours in the bundled 200-d / 300-d tables |
| `05_toy_attention.py` | Raffel attention recovering a planted peak hidden state |
| `06_bow_baseline.py` | Multinomial NB control, with a hashtag-stripped ablation |
| `07_average_pooling.py` | The sklearn-stack feature: mean GloVe-like vectors, including zeros |
| `08_results_table.py` | Pretty-print `docs/metrics/published_metrics.csv` |

```bash
python3 examples/01_explore_dataset.py
python3 examples/01_explore_dataset.py --markdown
python3 examples/02_emoji_cooccurrence.py --split train --top 15
python3 examples/03_lexical_cues.py
python3 examples/04_inspect_emoji2vec.py --query 😒 😑 ❤
python3 examples/05_toy_attention.py --steps 5 --peak 2
python3 examples/06_bow_baseline.py
python3 examples/07_average_pooling.py
python3 examples/08_results_table.py
```

`examples/common/` is a tiny library:

* `io.load_split` — line-oriented CSV reader matching this repo’s files
* `tokenize.tokenize_tweet` — comma-to-space then a tweet-ish regex
* `emoji.emoji_code_points` — pictograph extractor without PyPI `emoji`
* `word2vec.load_word2vec_binary` — Gensim-less reader for the `.bin` files
* `attention.AdditiveAttention` — NumPy twin of `attention_layer.py`
* `bow.CountVectorizer` / `MultinomialNB` — the control classifier

Original 2023 training code is unchanged. These examples are for reading
the dataset and for checking that the documented claims about splits /
emoji coverage / attention mechanics are actually true.
