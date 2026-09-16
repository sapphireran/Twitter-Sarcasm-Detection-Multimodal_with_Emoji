# Twitter sarcasm detection with emoji embeddings

Personal 2023 University of Copenhagen Computational Cognitive Science 2
final project. The code in this repository is course work, not company
work.

**Question:** if you already have Twitter-trained GloVe vectors, does
adding `emoji2vec` help a sarcasm classifier?

**Recorded answer:** a little on the balanced 2,000-tweet test set, more
on the 278-tweet emoji-heavy subtest. The deep model is the one to cite:

| Setting | Accuracy | F1 |
| --- | ---: | ---: |
| Bi-LSTM + attention, word only, full test | 0.8635 | 0.8656 |
| Bi-LSTM + attention, word + emoji, full test | **0.8735** | **0.8686** |
| Same model, word + emoji, emoji subtest | 0.8921 | 0.9107 |

Full tables, including SVM / trees / forests, are in
[docs/results.md](docs/results.md). Those numbers are copied from the
committed notebook outputs. They are not a fresh eval — GloVe Twitter
27B and most trained weights are not in git.

## What is in this repo

```
attention_layer.py     Raffel-style attention over LSTM states
data_utils.py          tweet load, mean-pool, Keras padding
dl_model.py            Bi-LSTM + attention builder
dataset/               train (39,780) / test (2,000) / subtest (278)
emoji2vec_twitter.bin  emoji channel used by the notebooks
docs/                  project write-up
examples/              walkthroughs that run without GloVe or TF
```

The two views the notebooks call **W** and **WE**:

- **W** — 200-d GloVe Twitter vectors only. OOV emoji become zeros.
- **WE** — classical models concatenate a 200-d emoji2vec average
  (400-d tweets). The deep model instead writes emoji2vec rows into
  the same 200-d embedding table for OOV tokens that contain emoji.

There is no image model. "Multi-modal" here means two text embedding
tables.

## Read the docs

| Page | Contents |
| --- | --- |
| [docs/project-overview.md](docs/project-overview.md) | Research question and layout |
| [docs/dataset.md](docs/dataset.md) | Splits, tags, emoji rates, examples |
| [docs/architecture.md](docs/architecture.md) | Mean-pool baselines vs Bi-LSTM |
| [docs/code-map.md](docs/code-map.md) | Function-level notes |
| [docs/results.md](docs/results.md) | Accuracy / F1 / recall / precision |
| [docs/reproduction.md](docs/reproduction.md) | How to replay the 2023 notebooks |

## Run the examples (no GloVe required)

```bash
python3 examples/inspect_dataset.py
python3 examples/preprocess_walkthrough.py
python3 examples/emoji_signal.py
python3 examples/heuristic_baseline.py
python3 examples/attention_demo.py
python3 examples/report_metrics.py
python3 examples/run_all.py
python3 -m unittest discover -s examples/tests -v
```

Details: [examples/README.md](examples/README.md).

The original notebooks need TensorFlow 2.10-era Keras, Gensim 3,
NLTK, and `glove.twitter.27B.200d.bin`. See
[docs/reproduction.md](docs/reproduction.md) before you try to
`load_model` the graphs under `model/`.

## License

MIT. See [LICENSE](LICENSE).
