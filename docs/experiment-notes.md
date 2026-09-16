# Experiment notes (personal, June 2023)

These are the decisions I would have to reconstruct from the
notebooks if I opened this repo cold. Written in 2026 while expanding
the personal docs; not part of the original hand-in.

## Why emoji at all

Sarcasm on Twitter in this dump is often a **positive sentence plus a
disclaiming tail**: `#not`, `#yeahright`, `😒`, `😑`. Word-only GloVe
already sees `#not` as a token. The bet was that *emoji identity*
still helps when the tail is a face and not a hashtag, especially if
that face lives in a vector space aligned with Twitter GloVe
(emoji2vec).

The official test set only half-confirms the bet (+1.0 accuracy on
the BiLSTM, +0.35 on forest). The emoji-rich subtest confirms it more
loudly. That is why both columns stayed in the write-up.

## Two views, one shuffle

`ml_read_data` builds W and WE features and then applies the same
permutation to both. That was deliberate: I did not want a situation
where SVM-W and SVM-WE were compared on differently shuffled rows
while I was still debugging accuracy prints. There is still no seed,
so a later re-fit is a new shuffle.

## Frozen embeddings

`PrepModel` sets `trainable=False` on the embedding layer. With
39k tweets I did not want 200 × vocab parameters moving on top of
two 256-d BiLSTMs. The WE vs W comparison is then a clean statement
about the *initial* matrix (emoji fallback on or off), not about
which rows the optimiser decided to nudge.

## Attention instead of a final LSTM state

The first draft was a single BiLSTM without `return_sequences`. It
under-used late hashtags. Switching to `return_sequences=True` twice
and adding the Raffel layer (the `Attention` class already lived in
the course materials / Keras 2 folklore) was the change that moved
validation accuracy into the mid-80s. I did not sweep number of
heads or replace it with additive attention; one vector `W` was
enough for 78 steps.

## Subtest is not a second official test

I carved the 278-row slice because I wanted a place where "the tweet
actually has an emoji" was the common case. It is more sarcastic
(62%) than the balanced test, so a dummy "always sarcastic" classifier
already gets 62% there. Report subtest next to test, never instead of
it.

## Things I would not repeat

* **No `requirements.txt` and no seed.** That is why this snapshot
  can replay the *ideas* but not the *bytes*.
* **Mixed Keras imports** in `dl_model.py`
  (`tensorflow.keras` *and* `tensorflow.python.keras`). That is why
  the saved summaries are full of `ModuleWrapper_*`.
* **Except-blocks that train the wrong estimator.** If a DT / RF /
  GBT pickle is missing, several cells construct `SVC()`. Harmless
  when the pickle exists; disastrous if someone "just runs the
  notebook" on a clean clone.
* **`Preprocess` sizes the embedding matrix with `len(lines)`**
  rather than `len(word_index)+1`. It works on this dump. It is
  still the wrong shape formula.
* **Metrics notebook vs baseline notebook pickle names**
  (`svm_model.pkl` vs `svm_classifier.pkl`). I must have renamed
  files locally and not updated both notebooks.

## What the new examples are for

I do not want the only runnable artefact to be a 2023 Keras snapshot
that cannot load. The `examples/` scripts are the version of this
project I can open on a fresh machine: they prove the CSVs still
align, they show why `#not` is not the whole story, and they let the
attention equation be inspected without TensorFlow.
