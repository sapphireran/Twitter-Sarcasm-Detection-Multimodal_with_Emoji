# Examples

The scripts under `examples/` are the hands-on companion to the write-up.
They are personal study notes in code form: each one answers a question
you would otherwise have to reconstruct from notebook cells.

| Script | Question it answers | Extra files |
| --- | --- | --- |
| `01_dataset_preview.py` | How big are the splits, and what does a row look like? | CSVs only |
| `02_lexical_cues.py` | How far do hashtags and emoji presence get you? | CSVs only |
| `03_emoji_vectors.py` | What did we actually store in `emoji2vec_twitter.bin`? | 200-d bin |
| `04_attention_walkthrough.py` | What does `Attention.call` do to a short sequence? | none |
| `05_tfidf_baseline.py` | What accuracy is possible with bag-of-tokens on a laptop? | CSVs only |

Shared helpers live in `examples/lib/`. Tests in `tests/` import those
helpers; they do not scrape script stdout.

## Running

From the repository root (or anywhere — paths are relative to the file):

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/01_dataset_preview.py
python3 examples/02_lexical_cues.py
python3 examples/03_emoji_vectors.py
python3 examples/04_attention_walkthrough.py
python3 examples/05_tfidf_baseline.py
```

Every script uses only the standard library plus NumPy. None of them call
a social-media API.

## 01 — dataset preview

Loads each split with `examples.lib.io.load_split`, which pairs line `i`
of the sentence file with line `i` of the label file. It prints:

- row counts and class balance
- emoji / hashtag coverage
- three short labelled examples per split

Use this when you want to confirm the CSVs survived a clone.

## 02 — lexical cues

Computes, per split:

- coverage and sarcastic precision of `#not`, `#sarcasm`,
  `#sarcastictweet`, `#yeahright`, and `#sarcastic`
- the same two numbers for "tweet contains any emoji"
- a tiny rule baseline: predict sarcastic if any of those tags is present

This is the hashtag-leakage demonstration discussed in
[`dataset.md`](dataset.md). On the test set the tag rule is high precision
and incomplete recall. That is the shape you want to remember before you
read a 87% neural score.

## 03 — emoji vectors

Parses the word2vec binary header (`1661 200`) and the 1,661 rows. It
then:

- looks up 😂, 😒, 😍, and ❤️
- prints each vector's L2 norm
- lists the five nearest in-file neighbours by cosine similarity

The neighbour lists are a sanity check that the file is a real embedding
and not a truncated upload. They are **not** a linguistic analysis of
sarcasm.

A second block averages the emoji present in a few test tweets and prints
the cosine between those tweet-level emoji means. Tweets that share 😒
sit closer than a 😒 tweet and a ❤ tweet.

## 04 — attention walkthrough

Builds a batch of shape `(2, 4, 3)`:

- row 0: filler, filler, **cue**, filler
- row 1: the same steps, but we mask the cue

It uses a `W` aligned with the cue direction so the unmasked row puts most
of its mass on step 3. The masked row cannot, and the pooled vector moves
toward the filler. This is the same arithmetic as `attention_layer.py`,
including `tanh`, the additive bias, post-exp masking, and `epsilon`.

## 05 — TF-IDF baseline

Fits a hashed-unigram logistic model on the training CSV:

1. Tokenise with the example tweet tokenizer (hashtags, mentions, emoji
   kept intact).
2. Hash tokens into a fixed-width bag (default 4,096).
3. Weight by a smoothed inverse-document frequency computed on train.
4. Train L2-regularised logistic regression with mini-batches.
5. Evaluate accuracy and F1 on test and subtest.
6. Repeat after dropping tokens that start with `#`.

The hash trick keeps the example off sklearn and off a 40k-by-vocab dense
matrix. It will not match the GloVe SVM. It *will* show that surface
tokens, especially hashtags, already explain a large fraction of the
label.

Optional flags:

```bash
python3 examples/05_tfidf_baseline.py --hash-size 2048 --epochs 4 --seed 0
python3 examples/05_tfidf_baseline.py --max-train 8000
```

## Library map

| Module | Responsibility |
| --- | --- |
| `examples/lib/io.py` | repo root, split loading, label validation |
| `examples/lib/tokenize.py` | regex tweet tokenizer + emoji / hashtag finders |
| `examples/lib/word2vec_bin.py` | Gensim-free word2vec `.bin` reader |
| `examples/lib/attention.py` | NumPy `tanh` attention |
| `examples/lib/hashed_tfidf.py` | hash bag, IDF, logistic regression |

## What these examples deliberately skip

- No TensorFlow model load. The SavedModel trees are incomplete here.
- No GloVe download. That file is huge and is not required to understand
  the project.
- No live Twitter / X calls. The course data is already on disk.
