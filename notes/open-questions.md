# Open questions I still have

Personal leftover list. None of these are answered by a file in the repo. I am writing them down so I do not pretend the 2023 report closed the topic.

## Is subtest exactly `test ∩ emoji`?

The class counts on emoji-bearing test rows (171 sarcastic / 106 not) match subtest's emoji-bearing rows exactly. Subtest has 278 rows and 277 emoji hits with my regex, so either:

- subtest is that subset plus one sarcastic tweet my regex missed, or
- I built a separate emoji-heavy draw that happened to land on the same 171 / 106 split (unlikely).

I should `diff` the two sentence files instead of guessing. If they match, every "subtest" number in the experiment log is just "full test, but only the rows where the extra channel exists." That is a cleaner paper sentence than "a second test set."

## Are the distant-supervision labels any good?

`Happy birthday to me. Yay.` is sarcastic in train. Maybe. `Late nights early mornings` repeats with different faces and mixed labels. I never measured label noise. A 200-row hand check would tell me whether 87% acc is close to the annotation ceiling.

Related: I do not know the original hashtag rule (`#sarcasm`, `#not`, …) or whether test was hand-checked. The files arrived as CSVs.

## Why did SVM get worse on WE / full test?

Three guesses I did not test:

1. Most WE vectors are `[glove; ~0]`. The extra 200 zeros change the RBF (or whatever default kernel) geometry enough to hurt.
2. The unseeded shuffle in `ml_read_data` is called separately for train and test, so the *data* is the same but I have no paired residual plot from a single transform.
3. Default `C=1` is fine in 200-d and slightly wrong in 400-d.

A linear SVM vs RBF, and a run that *only* concatenates the emoji half when it is non-zero, would separate (1) from (3).

## Is the deep-model WE win just "fewer zero rows"?

W already has a timestep for 😂; it is often a zero vector. WE fills that slot. An ablation that uses a single learned "unk-emoji" vector instead of emoji2vec would tell me whether I needed Eisner / emoji2vec at all, or just a dedicated unk.

I also never checked how many train vocab items actually took the emoji2vec branch. `nf` is incremented in `Preprocess` and never printed.

## Attention

If I open this project again, first plot: mean attention mass on emoji timesteps, W vs WE, for the 277 emoji test rows. If WE does not move that mass, the +1 acc is coming from something duller (a slightly different embedding neighborhood for a handful of tokens) and I should say so.

## Calibration

WE has higher full-test acc and higher BCE (0.320 vs 0.312). That smells like a more decisive, worse-calibrated model (precision 0.90, recall 0.84). Reliability diagrams would make that a real claim.

## Domain shift I ignored

`<user>` appears in 9,433 train rows and zero test / subtest rows. Hashtag rate is also higher on test (932 / 2000) than a naive 13% of train would suggest — wait, train has 8,486 / 39,780 ≈ 21% hashtag rows, test 932 / 2000 = 47%. That is a real shift. I never trained a "hashtag-aware" split or stratified anything.

## Compute / seed

No seed, no epoch log, no batch size. The two snapshot names are the whole training diary. A rerun that lands within ~1 acc point of 0.874 WE is success; bit-identical is fantasy.

## What I would run in one weekend

1. Confirm subtest ⊆ test with a script.
2. Hand-label 200 test rows; estimate noise.
3. Forest and BiLSTM on `test ∩ emoji` only (if that is not already subtest).
4. Attention-on-emoji plot.
5. One frozen-embedding transformer so I know the 2023 ceiling.
6. Fix the three `SVC()` except-blocks before anyone retrains.

That is enough. I do not need a new architecture to finish the sentence I started in 2023: emoji helps a little, and only when it is there.
