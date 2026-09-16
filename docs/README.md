# Docs index

Write-up of the personal CCS2 2023 sarcasm project. Read these in order if you are new to the snapshot; jump in if you already know the notebooks.

| Order | Page | One-line job |
| ---: | --- | --- |
| 1 | [dataset.md](dataset.md) | What is in `dataset/`, why `subtest` exists, cue-tag leakage. |
| 2 | [preprocessing.md](preprocessing.md) | Comma collapse, TweetTokenizer, mean-pool vs pad. |
| 3 | [embeddings.md](embeddings.md) | GloVe Twitter 200-d vs `emoji2vec_twitter.bin` (200-d) vs `emoji2vec.bin` (300-d). |
| 4 | [architecture.md](architecture.md) | sklearn concat vs Bi-LSTM + Raffel attention. |
| 5 | [training.md](training.md) | Fit recipe, missing GloVe, DT/`SVC()` typo. |
| 6 | [evaluation.md](evaluation.md) | 2023 tables + live cue baseline. |
| 7 | [notebooks.md](notebooks.md) | Path differences between the three `.ipynb`s. |
| 8 | [references.md](references.md) | Papers the code actually depends on. |

Runnable companions live in [`../examples/`](../examples/README.md). They stay on the standard library plus NumPy so they run without GloVe or TensorFlow.
