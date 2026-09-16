# Personal examples

These scripts walk the 2023 UCPH Computational Cognitive Science 2 sarcasm
project **without** TensorFlow, Gensim, NLTK, or the 1.2-million-word
Twitter GloVe dump. They read the CSVs that already live in `dataset/`
and replay the control flow of `data_utils.py` / `attention_layer.py`
with dummy vectors and a NumPy attention layer.

Run everything from the repository root. The only extra dependency is
NumPy (already used by the original pipeline).

```bash
python3 examples/explore_dataset.py
python3 examples/tokenize_tweets.py --split subtest --n 4
python3 examples/sarcasm_cues.py --split test
python3 examples/attention_numpy.py
python3 examples/embedding_pipeline.py --split subtest --n 6
python3 examples/report_results.py --all
python3 -m unittest tests.test_examples
```

## What each script is for

| Script | What it shows |
| --- | --- |
| `explore_dataset.py` | Split sizes, label balance, token / emoji / hashtag rates |
| `tokenize_tweets.py` | Tweet-ish tokenizer plus the 2023 `ReadOpen` comma-join quirk |
| `sarcasm_cues.py` | Transparent `#not` / contrast rule vs. the official labels |
| `attention_numpy.py` | Raffel-style attention weights on a toy sequence and a tweet |
| `embedding_pipeline.py` | Mean-pooling, 2× concat, Keras-style padding on real tweets |
| `report_results.py` | The recorded June 2023 accuracy / F1 / precision / recall tables |

`examples/lib/` is the shared library behind those CLIs. It is
intentionally small and dependency-light so the write-ups in `docs/`
can point at runnable code instead of notebook screenshots.

The original training notebooks (`baseline_models.ipynb`,
`get_metrics_of_models.ipynb`, `evaluate_loaded_dl_models.ipynb`) still
need the missing GloVe binary and a Keras 2 / TensorFlow 2 environment.
See [docs/reproduction.md](../docs/reproduction.md).
