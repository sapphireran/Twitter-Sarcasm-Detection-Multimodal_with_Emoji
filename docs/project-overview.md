# Project overview

This is a personal 2023 University of Copenhagen (UCPH) Computational
Cognitive Science 2 final project. The question is whether **emoji
embeddings** add anything useful on top of **Twitter-trained word
embeddings** for binary sarcasm detection.

The short answer from the original notebooks: yes, but the gain is
uneven. The Bi-LSTM + attention model already does most of the work
from text. Emoji vectors help most on the emoji-heavy **subtest**,
where every tweet contains non-ASCII characters.

This document is a map of the personal research code in this
repository. It does not re-train the original models. Those notebooks
were written against TensorFlow / Keras 2.x, NLTK, Gensim, and a
Twitter GloVe file that is **not** checked in (it is several
gigabytes).

## What "multi-modal" means here

There is no image or audio stream. Both channels are still text:

| Channel | Source | Typical size | Role |
| --- | --- | --- | --- |
| Word / tweet text | GloVe Twitter 27B, 200-d | 200 | Main lexical signal |
| Emoji | `emoji2vec` / `emoji2vec_twitter` | 200 | Extra signal when a token is an emoji or contains one |

Classical models concatenate the two averaged vectors into a **400-d**
tweet embedding. The deep model keeps a single 200-d token embedding
table: known words get GloVe rows, unknown tokens that contain emoji
get an averaged `emoji2vec` row instead of zeros.

That is why the notebooks label runs as:

- **W** — word embeddings only
- **WE** — word embeddings plus emoji embeddings

## Research question

Sarcasm on Twitter is often marked by:

1. Polar-opposite wording ("I *love* waiting three hours").
2. Explicit tags such as `#not`, `#sarcasm`, `#sarcastictweet`.
3. Emoji that flip or amplify the tone (😒 after a positive clause).

The project tests whether (3) is recoverable from pretrained emoji
vectors once (1) and (2) are already available to a classifier.

The **subtest** (278 tweets) is the diagnostic split: every line has
non-ASCII characters, and 129 of 172 sarcastic tweets carry an
explicit sarcasm hashtag. If emoji vectors are doing real work, the
W → WE gap should be larger there than on the balanced 2,000-tweet
test set. That is what the recorded numbers show for Random Forest
and Bi-LSTM + attention.

## Repository layout

```
attention_layer.py          Raffel-style temporal attention (Keras)
data_utils.py               Tweet load, average embeddings, Keras pad
dl_model.py                 Bi-LSTM + attention builder
baseline_models.ipynb       SVM / DT / RF / GBT train-or-load
evaluate_loaded_dl_models.ipynb   Load saved Keras models
get_metrics_of_models.ipynb Accuracy / F1 / recall / precision
dataset/                    train / test / subtest sentence+label CSVs
baseline_models/            pickled sklearn models (partial)
model/                      saved Keras graphs (weights may be incomplete)
emoji2vec.bin               generic emoji2vec
emoji2vec_twitter.bin       Twitter-tuned emoji2vec used in notebooks
docs/                       this documentation
examples/                   runnable walkthroughs (stdlib + numpy)
```

## What you can run today

The original training notebooks need GloVe Twitter 27B (binary or
text), NLTK `TweetTokenizer`, Gensim `KeyedVectors`, scikit-learn,
and a Keras 2.x stack. Those are documented in
[reproduction.md](reproduction.md).

The `examples/` scripts do **not** need that stack. They:

- inspect the checked-in CSVs
- replay the project's tokenizer / padding logic
- measure emoji and hashtag cues against labels
- implement the attention scoring math in NumPy
- reprint the original notebook metrics as tables

See [../examples/README.md](../examples/README.md).

## Related reading

- Raffel, Luong, Liu, Johnson, and Kiros. *A Common Architecture for
  Natural Language Generation*. Feed-forward attention over RNN
  states: https://arxiv.org/abs/1512.08756
- Eisner, Rocktäschel, Augenstein, Bošnjak, and Riedel. *emoji2vec:
  Learning Emoji Representations from their Description*.
  https://arxiv.org/abs/1609.08359
- Pennington, Socher, and Manning. *GloVe: Global Vectors for Word
  Representation*. The Twitter 27B 200-d vectors are the word channel
  used in the notebooks.
