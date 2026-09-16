# Overview

This repository is the 2023 Computational Cognitive Science 2 final
project at the University of Copenhagen. The research question was
narrow:

> Does adding emoji co-occurrence vectors to tweet text embeddings
> help sarcasm detection, and if so, where?

Two feature families were compared:

* **W (word-only).** Each tweet is a 200-dimensional mean of GloVe
  Twitter 27B vectors, or a padded token sequence for the LSTM.
* **WE (word + emoji).** The same word representation plus a 200d mean
  of `emoji2vec_twitter.bin`, concatenated for the sklearn models, or
  mixed into the embedding matrix for the LSTM (emoji tokens that miss
  GloVe are filled from emoji2vec when `get_emoji2vec=True`).

The hypothesis at the time was that sarcasm on Twitter is often a
*clash* between a positive sentence and a deadpan or eye-roll emoji,
so a model that can see both channels should beat text alone.

The 2023 numbers support a weaker, more local claim:

1. The BiLSTM + attention WE model is about **+1.0 accuracy point** on
   the official 2,000-tweet test set (0.8735 vs 0.8635).
2. The same WE model is about **+2.5 points** on subtest (0.8921 vs
   0.8669).
3. Subtest is not a held-out domain. It is the emoji-bearing slice of
   test. WE can only help on tweets that contain emoji, so the larger
   subtest gap is the expected place to look, not a second dataset.

The archive lab on this branch exists to make those three facts
checkable from the CSVs and notebook outputs without TensorFlow.

## Course framing

CCS2 asked for a multi-modal cognitive-science-flavored model, not a
leaderboard submission. The design choices match that brief:

* frozen embeddings (no fine-tuning GloVe)
* a published attention layer (Raffel et al. 2015) instead of a custom
  transformer
* classical baselines (SVM, trees, forests, boosting) on the same
  averaged vectors
* a dedicated emoji-only evaluation slice

Treat the recorded scores as a course snapshot from June 2023, not as
a claim that this architecture is still competitive with later
instruction-tuned classifiers.
