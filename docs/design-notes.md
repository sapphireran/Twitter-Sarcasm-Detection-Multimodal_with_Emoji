# Design notes (personal)

Why the 2023 code looks the way it does, and why this docs/examples pass does not “clean it up” into a different experiment.

## Two modalities, one string

Calling emoji a second modality is slightly generous: there is no second encoder and no image. The claim that *is* supported is narrower — **a second pretrained table, indexed by the same tokens, changes decisions when those tokens are emoji.** Concat (classical) and in-slot fallback (deep) are two fusion operators over that table. Late fusion, emoji-only models, and contextual LMs were out of scope for a CCS2 final.

## Why attention instead of a last hidden state

Sarcasm markers in this dump are often **right-edge** (`#not`, a face, `#sarcastictweet`) after a positive stem. A unidirectional last state can work; bidirectional + attention is the version that can put mass on the marker regardless of side. The shipped `Attention` bias is per-step, which is a bit odd (it can learn “prefer position 17”) but matches Raffel’s temporal scoring with a global `W`.

## Why a lexical baseline now

The original notebooks never published a bag-of-cues number. Without it, 87% test acc is hard to interpret on a corpus where `#not` is almost a label leak. The NumPy logistic model is intentionally dumb: if it scores in the mid-70s, the BiLSTM’s extra 10 points have to come from order and distributed features, not from discovering `#not`.

## Why NumPy clones instead of refactoring `data_utils.py`

`data_utils.py` is glued to gensim’s `.vocab`, Keras Tokenizer, and the `emoji` package. Rewriting it would change the artifact students submitted. The `examples/lib` package is a **specification** of the ideas (mean pool, Raffel attention, comma flatten) that can be unit-tested on a laptop. The 2023 modules stay frozen.

## Subtest is a lens, not a third world

It is tempting to talk about “in-domain” vs “emoji domain.” Subtest is a **filter of test**. Gains there mean “on the tweets that actually contain high codepoints,” not “on a freshly sampled emoji corpus.” Docs say this every time the table appears because it is the easiest result to over-claim.

## What I would do in a rewrite (and did not do here)

1. Seed everything; persist the embedding arrays.
2. `mask_zero=True` on Embedding, drop the per-step bias or tie it to features.
3. Drop the 48 leaked test strings.
4. Report bootstrap CIs on subtest.
5. Add an emoji-only mean-pool ablation.
6. Pin TF and sklearn in a lockfile; export a complete SavedModel.

Those are future personal-project chores, not silent diffs against the course drop.
