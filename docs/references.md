# References

Personal course-project notes, not a paper bibliography. Links are the
public artifacts the 2023 code actually cites or depends on.

## Papers the code points at

* Colin Raffel and Daniel P. W. Ellis, *Feed-Forward Networks with
  Attention Can Solve Some Long-Term Memory Problems*, 2015.
  [arXiv:1512.08756](https://arxiv.org/abs/1512.08756).
  Implemented in [`attention_layer.py`](../attention_layer.py) and
  reimplemented in [`examples/attention.py`](../examples/attention.py).
* Jeffrey Pennington, Richard Socher, and Christopher D. Manning,
  *GloVe: Global Vectors for Word Representation*, EMNLP 2014.
  [Project page](https://nlp.stanford.edu/projects/glove/). The
  notebooks use the Twitter 27B, 200-d pretrained vectors.
* Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, and
  Sebastian Riedel, *emoji2vec: Learning Emoji Representations from
  their Description*, 2016. [arXiv:1609.08359](https://arxiv.org/abs/1609.08359).
  The repo vendors `emoji2vec.bin` and a Twitter-aligned
  `emoji2vec_twitter.bin`.

## Course context

* University of Copenhagen, Computational Cognitive Science 2, 2023
  final project.
* Title used on the GitHub repo: *Multi-modal Sarcasm Detection Using
  Textual Contents and Emoji Co-occurrences in Twitter*.
* Author / copyright holder on [`LICENSE`](../LICENSE): `pang990801`
  (Sapphire Ran), MIT License, 2023.

## Libraries the original files import

* TensorFlow / Keras 2 — `dl_model.py`, `attention_layer.py`, the
  evaluation notebooks
* `keras_preprocessing` — `Tokenizer`, `pad_sequences`
* Gensim `KeyedVectors` — GloVe and emoji2vec loaders
* NLTK `TweetTokenizer`
* `emoji` — glyph extraction inside `Preprocess`
* pandas, NumPy, scikit-learn, joblib

## Related files in this repo

| Want | Open |
| --- | --- |
| Split sizes and emoji rates | [dataset.md](dataset.md), `python examples/dataset_overview.py` |
| Feature construction | [preprocessing.md](preprocessing.md) |
| Model graphs | [architecture.md](architecture.md) |
| 2023 scores | [results.md](results.md) |
| How to run anything | [reproduction.md](reproduction.md) |
| Notebook-free walkthroughs | [examples/README.md](../examples/README.md) |
