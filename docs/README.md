# Documentation index

Personal write-up for the 2023 UCPH CCS2 multimodal Twitter sarcasm project. Nothing here is company code. The original experiment still lives in the repo-root `.py` / `.ipynb` files; this folder explains them.

Start at the [root README](../README.md) if you want the one-page version.

| Page | What it answers |
| --- | --- |
| [methodology.md](methodology.md) | What is being classified, and why a second vector channel exists |
| [data-pipeline.md](data-pipeline.md) | How a tweet line becomes tokens, averages, or padded ids |
| [emoji-fusion.md](emoji-fusion.md) | Concatenate-means vs mixed embedding table |
| [models.md](models.md) | SVM / trees / forest / GBT / Bi-LSTM + attention |
| [results.md](results.md) | The 2023 metric tables and how to read the multimodal lift |
| [api-reference.md](api-reference.md) | Functions in `data_utils.py`, `dl_model.py`, `attention_layer.py` |
| [notebooks.md](notebooks.md) | What each `.ipynb` actually runs |
| [reproduction.md](reproduction.md) | Examples path vs original-notebook path |
| [known-issues.md](known-issues.md) | Fallback bugs, path mismatches, Gensim 4, SavedModel gaps |

Runnable companions sit in [`examples/`](../examples/README.md). They reimplement the averaging, fusion, and attention math with deterministic hash embeddings so you do not need GloVe or TensorFlow to see the idea work.
