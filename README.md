# Multimodal Twitter sarcasm detection (text + emoji)

Personal final project for **UCPH Computational Cognitive Science 2
(2023)**. The question is whether emoji embeddings add anything on top
of word embeddings when you classify a tweet as sarcastic or not.

This repo is the original coursework code plus a later docs/examples
layer so the pipeline is readable without opening the notebooks.

## What is in here

Two families of models share the same tweet files:

1. **Classical baselines** — SVM, decision tree, random forest, gradient
   boosting. Each tweet is a single 200-d mean GloVe vector
   (*single-modal*) or that vector concatenated with a 200-d mean
   emoji2vec vector (*multi-modal*, 400-d).
2. **Bi-LSTM + attention** — token sequences go through a frozen 200-d
   embedding matrix, two bidirectional LSTMs (256 units each
   direction), Raffel-style attention, and a sigmoid. The multi-modal
   variant writes emoji2vec rows into the same embedding matrix when a
   token is an emoji rather than a GloVe word.

On the held-out test set (2,000 tweets, balanced), the original
notebooks report:

| Model | Text only (acc / F1) | Text + emoji (acc / F1) |
| --- | --- | --- |
| Decision tree | 0.727 / 0.756 | 0.730 / 0.757 |
| SVM | 0.769 / 0.772 | 0.763 / 0.766 |
| Gradient boosting | 0.746 / 0.751 | 0.748 / 0.753 |
| Random forest | 0.815 / 0.823 | 0.818 / 0.826 |
| Bi-LSTM + attention | **0.864 / 0.866** | **0.874 / 0.869** |

Emoji helps most on the small **emoji-heavy subtest** (278 tweets,
almost every row has an emoji): Bi-LSTM + attention goes from 0.867
accuracy / 0.894 F1 to **0.892 / 0.911**.

Full tables, precision, and recall: [docs/results.md](docs/results.md).

## Layout

```
attention_layer.py     Raffel et al. temporal attention (Keras)
data_utils.py          tweet load, mean-pool, Keras tokenizer + embedding matrix
dl_model.py            Bi-LSTM + attention builder
dataset/               train / test / subtest sentences + 0/1 labels
emoji2vec*.bin         precomputed emoji embeddings (included)
baseline_models/       pickled sklearn trees / GBT (SVM and RF pickles are not in git)
model/                 SavedModel graphs from the 2023 run (weights folder missing)
*.ipynb                original training / eval notebooks
docs/                  write-up of data, models, results, reproduction
examples/              runnable scripts that do not need GloVe or TensorFlow
tests/                 unit tests for the example libraries
```

## Quick start (docs examples)

The example scripts only need NumPy. From the repo root:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/inspect_dataset.py
python3 examples/tokenize_tweets.py
python3 examples/toy_embedding_fusion.py
python3 examples/attention_walkthrough.py
python3 examples/lexical_sarcasm_baseline.py
python3 -m unittest discover -s tests -v
```

`inspect_dataset.py` reprints split sizes, class balance, emoji /
hashtag rates, and a few labeled rows.
`lexical_sarcasm_baseline.py` trains a from-scratch logistic regression
on surface cues (`#not`, `#sarcasm`, elongation, emoji counts) so you
can see how much of the test set is solvable without embeddings.

## Reproduce the original notebooks

That path needs files that are **not** all in this git tree:

- GloVe Twitter 27B 200-d, converted to word2vec binary
  (`glove.twitter.27B.200d.bin`). Not committed (≈1–2 GB).
- `nltk` tweet tokenizer models.
- TensorFlow 2 with Keras 2-style `lr=` on `Adam` and
  `tensorflow.python.keras` imports.
- The SavedModel `variables/` directories — only `saved_model.pb` and
  `keras_metadata.pb` are present, so `tf.keras.models.load_model`
  will not restore weights as-is.

See [docs/reproduction.md](docs/reproduction.md) for the exact
notebook order, expected metrics, and known code quirks (wrong
fallback classifiers in `baseline_models.ipynb`, path mismatches,
gensim `.vocab` deprecation).

## Data in one paragraph

`dataset/train_sentence.csv` has 39,780 tweets (21,292 non-sarcastic /
18,488 sarcastic). `test_sentence.csv` is 2,000 tweets, exactly
balanced. `subtest_sentence.csv` is 278 tweets chosen so that 277
contain emoji. Labels are a parallel `*_label.csv` of `0` / `1` with no
header. Mentions are already normalized to `<user>`. A large slice of
the sarcastic test rows are tagged `#not` or `#sarcasm` — treat those
hashtags as both a linguistic cue and a dataset artifact.

Details: [docs/dataset.md](docs/dataset.md).

## License

MIT. Copyright (c) 2023 pang990801.
