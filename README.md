# Multi-modal sarcasm detection (text + emoji)

Personal 2023 final project for **Computational Cognitive Science 2** at the
University of Copenhagen. The task is binary sarcasm detection on Twitter
posts, comparing a text-only (GloVe) pipeline with a multi-modal pipeline
that also uses [emoji2vec](https://github.com/uclnlp/emoji2vec) co-occurrence
vectors.

This repository still contains the original notebooks and saved models. The
`docs/` and `examples/` tree added later documents the dataset, the
architecture, and a **stdlib-only** walkthrough so you can inspect the data
and a cue baseline without downloading the 1.2 GB Twitter GloVe dump.

## Results (from the executed 2023 notebooks)

Held-out **test** (2,000 tweets, balanced) and emoji-heavy **subtest**
(278 tweets). `single` = GloVe only; `multi` = GloVe + emoji2vec.

| Model | Test acc (single / multi) | Test F1 (single / multi) | Subtest acc (single / multi) |
| --- | --- | --- | --- |
| SVM | 0.769 / 0.763 | 0.772 / 0.766 | 0.813 / 0.824 |
| Decision tree | 0.727 / 0.730 | 0.756 / 0.757 | 0.777 / 0.799 |
| Random forest | 0.815 / 0.818 | 0.823 / 0.826 | 0.806 / 0.853 |
| Gradient boosting | 0.746 / 0.748 | 0.751 / 0.753 | 0.795 / 0.795 |
| **BiLSTM + attention** | **0.864 / 0.874** | **0.866 / 0.869** | **0.867 / 0.892** |

Emoji vectors help most on the subtest, which is almost entirely tweets that
contain emoji (see [docs/dataset.md](docs/dataset.md)). Full precision/recall
tables: [docs/results.md](docs/results.md).

## Repository layout

```
attention_layer.py     Raffel-style Keras attention used by the BiLSTM
dl_model.py            Bidirectional LSTM + attention classifier
data_utils.py          NLTK tokenizer, GloVe / emoji2vec averaging
baseline_models.ipynb  SVM, trees, forests, gradient boosting
evaluate_loaded_dl_models.ipynb
get_metrics_of_models.ipynb
dataset/               train / test / subtest sentence+label CSVs
model/                 saved Keras BiLSTMs (single + multi)
emoji2vec*.bin         pre-trained emoji embeddings
sarcasm_toolkit/       stdlib helpers for docs/examples
examples/              runnable walkthroughs (no TF, no GloVe, no network)
docs/                  dataset, architecture, methodology, reproduction
```

## Docs / examples (no TensorFlow required)

```bash
python -m sarcasm_toolkit
python examples/01_inspect_dataset.py
python examples/04_lexicon_baseline.py
python examples/06_pipeline_walkthrough.py
python -m unittest discover -s tests -v
```

Start with [docs/examples.md](docs/examples.md) and
[examples/README.md](examples/README.md). The original training path is in
[docs/reproduction.md](docs/reproduction.md).

## Original experiment stack

The 2023 notebooks expect:

* TensorFlow 2 / Keras, `keras-preprocessing`
* Gensim `KeyedVectors` for GloVe Twitter 200d and emoji2vec
* NLTK `TweetTokenizer`, `emoji`, pandas, scikit-learn, joblib

`glove.twitter.27B.200d.bin` is **not** in git (too large). Place it next to
the notebooks if you want to retrain. `requirements.txt` lists that stack;
`requirements-examples.txt` is only pytest for the stdlib tests.

## License

MIT. Course project by pang990801 / Sapphire Ran, 2023.
