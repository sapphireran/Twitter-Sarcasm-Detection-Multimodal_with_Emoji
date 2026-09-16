# Documentation index

Personal notes and worked examples for this 2023 UCPH Computational
Cognitive Science 2 final project: multimodal Twitter sarcasm detection
using GloVe word vectors plus emoji2vec.

Start here, then open the page that matches what you need.

| Document | What it covers |
| --- | --- |
| [Dataset](dataset.md) | Split sizes, label balance, file format, hashtag/emoji cues |
| [Architecture](architecture.md) | Bi-LSTM + attention network and the classical baselines |
| [Emoji modality](emoji_modality.md) | How emoji vectors are built and concatenated with text |
| [Results](results.md) | Accuracy / F1 / precision / recall from the original notebooks |
| [Reproduction](reproduction.md) | How to rerun training, what files are missing, known caveats |
| [Notebooks](notebooks.md) | What each `.ipynb` actually does |
| [Annotated examples](annotated_examples.md) | Real tweets from the test split, labeled by cue type |

Runnable scripts live in [`../examples/`](../examples/README.md).
They do **not** need the original GloVe dump or TensorFlow.
