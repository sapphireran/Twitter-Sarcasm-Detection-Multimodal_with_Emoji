# Project overview

This note is a personal record of the 2023 Computational Cognitive Science 2
final project. The implementation lives in the Python modules and notebooks
at the repository root. The writing here is meant to make that work readable
without opening every notebook cell.

## Research question

Given a short tweet, can a classifier decide whether the author is being
sarcastic? If yes, does a dedicated emoji embedding help beyond a Twitter-
trained word embedding?

The working hypothesis was:

1. Token-level order matters. Averaging GloVe vectors throws away the setup /
   punchline structure that sarcasm often uses.
2. Emoji are not just rare words. A laughing face after a complaint is a
   different signal from the same face after a compliment, so a vector space
   trained on emoji descriptions (emoji2vec) might carry information that
   GloVe Twitter does not assign to those glyphs.
3. The gain from (2) should be largest on tweets that actually contain emoji.
   That is why a 278-row **subtest** was carved out: 99.6% of those rows
   contain at least one emoji, versus about 14% of the main test set.

## Task definition

- Input: one tweet string, already somewhat cleaned (user mentions are often
  the token `<user>`).
- Output: `1` sarcastic, `0` not sarcastic.
- Evaluation: accuracy, F1, precision, and recall on a 2,000-row balanced
  test set, plus the same metrics on the emoji subtest.

The project did not attempt author-level sarcasm (Rajadesingan et al. 2015
style behavioural features) or conversation context. Each tweet is scored
in isolation.

## Two modelling tracks

**Classical track.** `data_utils.ml_read_data` tokenises with NLTK's
`TweetTokenizer`, looks up each token in GloVe, averages the hits into a
200-d tweet vector, and optionally concatenates a 200-d average of emoji2vec
hits. The result is a 200-d or 400-d bag-of-embeddings. SVM, a decision
tree, a random forest, and gradient boosting were fit on that matrix.

**Neural track.** `data_utils.Preprocess` builds a Keras tokenizer and a
frozen 200-d embedding matrix. Tokens found in GloVe use that row. Tokens
that look like emoji (via the `emoji` package) use the mean of their
emoji2vec rows when the multi-modal flag is on; otherwise they stay zero.
`dl_model.PrepModel` then stacks two bidirectional LSTMs (256 units each
direction) with dropout and the custom `Attention` layer from
`attention_layer.py`.

The two tracks share the same CSV splits and the same two embedding files.
They do not share a vectoriser: the classical models never see token order,
and the neural model never sees a concatenated 400-d average.

## What the numbers suggested

On the main test set, emoji concatenation barely moved the classical models
(random forest 81.5% → 81.8%) and helped the BiLSTM by about one point
(86.4% → 87.4%). On the emoji subtest the same multi-modal BiLSTM reached
89.2% accuracy and 0.911 F1.

That pattern matches the hypothesis: if most test tweets have no emoji, a
multi-modal embedding cannot do much. When the evaluation set is restricted
to tweets that contain emoji, the extra channel shows up.

A separate, less flattering observation is that the test set is saturated
with explicit sarcasm hashtags (`#not` appears 465 times, all sarcastic).
Any model that can see surface tokens will treat those tags as nearly
perfect features. The neural model still beat a strong random forest, which
is the more interesting comparison once hashtag leakage is acknowledged.
[`docs/dataset.md`](dataset.md) spells out the leak.

## What this documentation pass adds

The 2023 upload had a two-line README and no runnable path that avoided
GloVe. This pass adds:

- a project-level README
- dataset, architecture, attention, results, and reproduction notes
- examples that inspect the CSVs, read `emoji2vec_twitter.bin`, replay the
  attention equations in NumPy, and train a laptop-scale TF-IDF baseline

It does **not** retrain the BiLSTM or claim new official scores. The tables
in [`baselines-and-results.md`](baselines-and-results.md) are copied from
the executed notebooks.

## Non-goals

- No company or production code.
- No live social-media API calls. All examples read local CSV files.
- No attempt to modernise the Keras 2 attention layer to Keras 3.
- No claim that emoji2vec is the right emoji model in 2026; it was the
  standard lightweight choice in the course year.
