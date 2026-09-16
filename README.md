# Twitter sarcasm detection (text + emoji)

Personal dump of a University of Copenhagen **Computational Cognitive
Science 2** (2023) final project: *Multi-modal Sarcasm Detection Using
Textual Contents and Emoji Co-occurrences in Twitter*.

The original training notebooks expected GloVe Twitter 200-d, Keras 2,
and Gensim 3. This checkout keeps those files as they were uploaded and
adds a docs + examples layer that you can run with NumPy only.

## What is in here

* **39,780 / 2,000 / 278** labeled tweets in [`dataset/`](dataset/)
  (train / test / emoji-bearing subtest).
* Mean-pooled **GloVe** vs **GloVe ⊕ emoji2vec** baselines (SVM, trees,
  boosting) in [`baseline_models.ipynb`](baseline_models.ipynb).
* A **BiLSTM + Raffel attention** classifier in [`dl_model.py`](dl_model.py)
  and [`attention_layer.py`](attention_layer.py).
* Recorded 2023 scores: **0.874** test accuracy and **0.892** subtest
  accuracy for the multi-modal deep model (see [docs/results.md](docs/results.md)).
* Walkthroughs that do not need GloVe or TensorFlow in [`examples/`](examples/).

```text
tweet ─► TweetTokenizer
         ├─ mean GloVe 200-d              ─► sklearn baselines
         ├─ mean GloVe ⊕ mean emoji2vec   ─► sklearn baselines
         └─ frozen 200-d embedding
              ─► Dropout ─► BiLSTM ─► BiLSTM ─► Attention ─► sigmoid
```

## Run the examples (no GloVe)

From the repository root, with NumPy installed (`pip install -r requirements-examples.txt`):

```bash
python examples/dataset_overview.py
python examples/tokenize_demo.py
python examples/embedding_demo.py
python examples/attention_demo.py
python examples/sarcasm_cues.py
python examples/toy_pipeline.py
python -m unittest discover -s tests -v
```

`toy_pipeline.py` is a 16-tweet version of the single-modal vs
multi-modal baseline experiment. `attention_demo.py` is a NumPy copy
of the Keras attention layer sitting on a readable `#not` example.

## Docs

| Page | Contents |
| --- | --- |
| [docs/dataset.md](docs/dataset.md) | File formats, label balance, emoji rates, `ReadOpen` quirk |
| [docs/preprocessing.md](docs/preprocessing.md) | Tokenize → mean-pool vs embedding matrix |
| [docs/architecture.md](docs/architecture.md) | Baselines, BiLSTM, attention equations |
| [docs/results.md](docs/results.md) | Accuracy / F1 / precision / recall from the notebooks |
| [docs/reproduction.md](docs/reproduction.md) | GloVe, Gensim 3 vs 4, what will not load |
| [docs/notebooks.md](docs/notebooks.md) | Cell-by-cell map of the three uploaded notebooks |
| [docs/quirks.md](docs/quirks.md) | Path mismatches, missing weights, copy-paste fallbacks |
| [docs/references.md](docs/references.md) | Raffel 2015, GloVe, emoji2vec, course note |
| [examples/README.md](examples/README.md) | How to run the walkthroughs |

## Original source files

| Path | Role |
| --- | --- |
| [`data_utils.py`](data_utils.py) | Read tweets, mean-pool, build the 200-d matrix |
| [`attention_layer.py`](attention_layer.py) | Keras 2 attention (Raffel & Ellis 2015) |
| [`dl_model.py`](dl_model.py) | `PrepModel`: embedding → 2× BiLSTM → attention → dense |
| [`emoji2vec_twitter.bin`](emoji2vec_twitter.bin) | 200-d emoji vectors used by the notebooks |
| [`baseline_models/`](baseline_models/) | Decision Tree and GBT pickles (SVM / RF were not uploaded) |
| [`model/`](model/) | SavedModel graphs without `variables/` weights |

## License

MIT, 2023, `pang990801` — see [LICENSE](LICENSE).
