# 08 — Control experiments (docs-layer, 2026)

The 2023 notebooks never stripped `#sarcasm` and never trained a bag-of-words
model. The scripts under `examples/` fill that gap without TensorFlow. The
numbers in this page are produced by those scripts on the shipped CSVs; they
are **not** a claim that the 2023 BiLSTM was wrong, they are a reading aid.

## Cue-hashtag rule (`examples/03_lexical_cues.py`)

Predict sarcastic iff the tokenized tweet contains one of

`#sarcasm`, `#sarcastic`, `#sarcastictweet`, `#not`, `#notreally`, `#yeahright`, `#shocker`.

| Split | Cue rate, literal | Cue rate, sarcastic | Rule accuracy | Rule F1 | Majority |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.44% | 19.67% | 62.43 | 32.74 | 53.52 |
| test | **0.00%** | **61.40%** | **80.70** | 76.08 | 50.00 |
| subtest | 0.00% | 77.91% | 86.33 | 87.58 | 61.87 |

On this export, **no literal test tweet carries a cue hashtag**, and 61% of
sarcastic test tweets do. A one-feature rule already beats the 2023 SVM
(76.9%) and is within a few points of random forest (81.45%) on test. That is
distant-supervision residue, not a modelling result.

Two consequences:

1. Test accuracy in [04-results.md](04-results.md) is partly “did the net
   notice `#sarcastictweet`?”. The BiLSTM at 86.35% is still above the 80.7%
   rule, so it is reading more than the tag — but the gap is 5.6 points, not
   36 points over chance.
2. Subtest is *more* cue-saturated (78% of sarcastic rows), *and* 100%
   emoji. The WE vs W lift on subtest (see the 2023 table) is the only
   comparison that can be about faces rather than hashtags, and even there
   the hashtag is often present too.

Literal `#not` does exist in **train** (“#not ready yet”). The test split
happened not to include those.

## Bag-of-tokens Naive Bayes (`examples/06_bow_baseline.py`)

Multinomial NB, 4,000 unigrams, Laplace smoothing, numpy only.

| Mode | Test acc | Test F1 | Subtest acc | Subtest F1 |
| --- | ---: | ---: | ---: | ---: |
| full (hashtags kept) | 83.85 | 85.39 | 84.89 | 88.89 |
| stripped (hashtags removed) | 75.60 | 77.07 | 72.30 | 77.55 |
| emoji_flag (`__HAS_EMOJI__` + `__E:…__`) | 83.20 | 84.88 | **87.77** | 90.66 |

Reading:

* **Full BoW already beats GloVe-average SVM (76.9%) and RF (81.45%) on
  test.** Discrete `#not` is a better feature than a 200-d mean that dilutes
  it. That is expected and is a reason the neural net, which can attend to a
  single token, beats the sklearn stack.
* **Stripping hashtags costs ~8 points on test and ~12 on subtest.** The
  task does not collapse to chance, so there *is* sentence-level signal.
  75.6% is the honest lexical floor for a unigram model on this test set.
* **Emoji identity features help on subtest (+2.9 acc vs full) and do not
  help on test** (83.20 vs 83.85). That is the same shape as the 2023 WE vs
  W table, obtained with a one-hot flag instead of emoji2vec. It is
  independent evidence that the subtest filter is doing its job.

## Variation selector mismatch (`examples/04_inspect_emoji2vec.py`)

The most common “OOV” pictograph in train under a naive code-point match is
`❤` (U+2764, df=428). emoji2vec stores the fully-qualified `❤️`
(U+2764 + U+FE0F). `examples.common.word2vec.lookup_emoji` tries both. The
2023 `emoji` package typically emits the fully-qualified form, so the
notebooks may have hit this row even when a raw `ord()` walk does not.

Skin-tone modifiers (🏻–🏿) and some text-style dingbats (`☺`, `✌`, `♥`)
remain genuinely missing after that fallback.

## What we did *not* re-run

GloVe SVM / RF / BiLSTM. The weight shards and the 200-d GloVe file are not
in git. Do not treat the BoW table as a replacement for [04-results.md](04-results.md).
