# Multimodal Twitter sarcasm detection (text + emoji)

Personal 2023 course project for **Computational Cognitive Science 2** at the University of Copenhagen. The question is whether a tweet is sarcastic, and whether **emoji embeddings** help once the text is already represented with Twitter GloVe vectors.

This repository is **personal academic work**. It is not affiliated with any employer.

## What is in this repo

| Path | Role |
| --- | --- |
| `dataset/` | Aligned tweet / label files for train, test, and an emoji-heavy subtest |
| `data_utils.py` | Tweet tokenization, mean-pooling, and Keras sequence preprocessing |
| `dl_model.py` | Bidirectional LSTM + attention classifier |
| `attention_layer.py` | Raffel-style temporal attention (Keras) |
| `baseline_models.ipynb` | SVM / tree / forest / GBT baselines (word vs word+emoji) |
| `evaluate_loaded_dl_models.ipynb` | Load saved Keras models and report accuracy |
| `get_metrics_of_models.ipynb` | Accuracy, F1, precision, recall tables |
| `emoji2vec.bin`, `emoji2vec_twitter.bin` | Pretrained emoji vectors used at training time |
| `model/` | Saved Keras graphs for the best single- and multimodal nets |
| `docs/` | Project write-up: data, architecture, metrics, how to reproduce |
| `examples/` | Scripts that run **without** GloVe or TensorFlow |

The original notebooks expect `glove.twitter.27B.200d.bin` (not committed; it is large) plus TensorFlow / Gensim / NLTK. The `examples/` tree is the path that works on a stock Python + NumPy machine.

## Results snapshot (from the 2023 notebooks)

On the **balanced 2,000-tweet test set**, the saved Bi-LSTM + attention model reached **86.35%** accuracy from text embeddings alone and **87.35%** when emoji2vec was mixed into the embedding table. The same multimodal model reached **89.21%** on the 278-tweet emoji-rich subtest.

Classical baselines that average GloVe / emoji2vec per tweet sit lower. Random forest was the strongest of those (about **81.5–81.8%** on the full test set). See [docs/results.md](docs/results.md) for the full table.

## Run the lightweight examples

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/dataset_overview.py
python3 examples/cue_analysis.py
python3 examples/lexical_baseline.py
python3 examples/attention_walkthrough.py
python3 examples/tokenize_demo.py
python3 -m unittest discover -s tests -v
```

These scripts tokenize the committed CSVs, measure how sarcasm cues (hashtags, emoji, stock phrases) line up with labels, train a bag-of-words Naive Bayes / SGD logistic baseline, and walk through the same attention equations used in `attention_layer.py`.

## Original deep-learning path

1. Obtain [GloVe Twitter 27B 200d](https://nlp.stanford.edu/projects/glove/) and convert it to word2vec binary, or point the notebooks at a text dump (`glove_tt.txt` is what `get_metrics_of_models.ipynb` used).
2. Install the stack in [docs/reproduction.md](docs/reproduction.md).
3. Keep tweet files next to the notebooks, or use the `dataset/` copies (the metrics notebook already prefixes `dataset/`).

## Documentation

- [Dataset](docs/dataset.md) — splits, labels, file format, caveats
- [Preprocessing](docs/preprocessing.md) — `ReadOpen`, mean vectors, Keras padding
- [Architecture](docs/architecture.md) — embeddings, Bi-LSTM, attention
- [Results](docs/results.md) — notebook metrics, single- vs multimodal
- [Reproduction](docs/reproduction.md) — environments, missing artifacts, known quirks
- [References](docs/references.md) — papers and pretrained vectors
- [Examples](examples/README.md) — how the new scripts map onto the project

## License

MIT. See [LICENSE](LICENSE).
