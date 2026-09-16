# 01 — Overview

## The modelling question

Sarcasm on Twitter is often a *mismatch* between the words and the affect.
A tweet that says “I love lectures at 8am 😒” is sarcastic because the
positive predicate is cancelled by the deadpan face (and, in many corpora,
by a `#not` / `#sarcasm` tag). A model that only sees word embeddings will
encode “love” as positive and may miss the cancellation unless the
pictograph has its own vector.

This project asks a narrow version of that question:

> Holding the architecture fixed, does injecting emoji2vec — either as
> extra dimensions on a tweet-level average, or as non-zero rows in a
> frozen embedding matrix — improve sarcasm detection, especially on tweets
> that actually contain emoji?

“Multimodal” here does **not** mean image pixels, speech, or a separate
emoji encoder with cross-attention. Both channels are 200-dimensional
vectors living in (approximately) the same GloVe-Twitter space. The emoji
table was trained by Eisner et al. (2016) so that an emoji lands near the
words in its Unicode name / description. `emoji2vec_twitter.bin` is a 200-d
projection of that idea, which is why it can be concatenated with
GloVe-Twitter 200d without a learned adapter.

## Two stacks, one dataset

```
                    ┌─ average GloVe rows (200-d) ──────── sklearn (SVM, DT, RF, GBT)
tweet ─ tokenize ───┤
                    └─ sequence of ids + frozen E ──────── BiLSTM + attention → sigmoid
                              │
                              └─ optional: emoji2vec rows for pictograph tokens
```

The sklearn stack (`ml_read_data`) throws away order. That is a harsh
inductive bias for sarcasm, which is often a reversal *in time* (“yeah,
great job” after a complaint). The neural stack keeps order and lets
attention put mass on the reversal and on the emoji.

The control is always the same architecture with emoji rows zeroed
(`get_emoji2vec=False`) or, for sklearn, a 200-d vector instead of 400-d.
That isolates the emoji channel from “we trained a bigger model”.

## What this archive is for

The original course drop was four Python files, three notebooks, two
emoji2vec binaries, and almost no prose. These `docs/` pages reconstruct
the experimental design from the notebooks’ outputs so the personal repo
is readable without opening a 2 MB `.ipynb` with embedded plot PNGs.

They do **not** re-claim the numbers as a new paper. The tables in
[04-results.md](04-results.md) are copied from executed notebook cells dated
2023-06-05 / 2023-06-06.
