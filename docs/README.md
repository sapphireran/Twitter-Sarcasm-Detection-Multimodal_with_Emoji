# Documentation index

Personal notes for the UCPH CCS2 2023 sarcasm-detection project. Start at the
root [`README.md`](../README.md) if you want the short version.

| Page | Contents |
| --- | --- |
| [dataset.md](dataset.md) | Split sizes, labels, hashtags, emoji, leakage |
| [preprocessing.md](preprocessing.md) | `ReadOpen`, averaged vectors, Keras sequences |
| [architecture.md](architecture.md) | sklearn stack vs BiLSTM + attention |
| [attention.md](attention.md) | Raffel-style pooling used in `attention_layer.py` |
| [models-and-results.md](models-and-results.md) | Transcribed notebook metrics |
| [reproducing-experiments.md](reproducing-experiments.md) | Missing files, path traps, env notes |
| [limitations.md](limitations.md) | What the 2023 numbers do not prove |

Runnable companions live in [`../examples/README.md`](../examples/README.md).
They use only the Python standard library and the CSVs already in `dataset/`.
