# Personal project docs

These notes belong to the 2023 UCPH CCS2 sarcasm-detection project. They
are written after the fact so the notebooks are not the only explanation
of what the code does.

| Page | Use it when you want to… |
| --- | --- |
| [dataset.md](dataset.md) | Trust the CSVs: sizes, balance, hashtags, emoji rates |
| [preprocessing.md](preprocessing.md) | Follow a tweet from disk to a 200-d / 400-d vector |
| [architecture.md](architecture.md) | See how the BiLSTM + attention stack is wired |
| [results.md](results.md) | Read the recorded test / subtest tables |
| [reproduction.md](reproduction.md) | Know what is (and is not) in this GitHub snapshot |
| [experiment-notes.md](experiment-notes.md) | See the personal decisions from June 2023 |
| [code-map.md](code-map.md) | Open files in a sensible order |

Runnable counterparts live in [`examples/`](../examples/README.md). Those
scripts are the preferred way to *check* a claim in these pages: they
read `dataset/` directly and do not need GloVe.
