# Experiments

Numbers on this page are copied from executed cells in `get_metrics_of_models.ipynb` and `evaluate_loaded_dl_models.ipynb` (June 2023). They are also stored in [`results/recorded_metrics.json`](results/recorded_metrics.json). Nothing here is a new training run.

## Protocol (as the notebooks actually did it)

1. Load GloVe-Twitter 200d and `emoji2vec_twitter.bin`.
2. **Sklearn:** `ml_read_data` → mean-pooled 200-d / 400-d features → predict with pickles under `baseline_models/`.
3. **Neural:** `ReadOpen` + `Preprocess` / `preprocess_test` → `tf.keras.models.load_model("model/best_model_*")` → `evaluate` and thresholded F1.
4. Report four numbers per metric: test word, test word+emoji, subtest word, subtest word+emoji.

Test is 2,000 tweets (1,000 / 1,000). Subtest is 278 tweets (172 / 106), almost all emoji-bearing. See [dataset.md](dataset.md).

Threshold for F1 / precision / recall on the neural net is the default `predict` > 0.5 path used in the metrics notebook (`astype` / `f1_score` on binarized outputs).

## Accuracy

| Model | Test word | Test word+emoji | Subtest word | Subtest word+emoji |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| Decision tree | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| Random forest | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| Gradient boosting | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| BiLSTM + attention | 0.8635 | **0.8735** | 0.8669 | **0.8921** |

Keras `evaluate` printed the same neural accuracies with a short loss:

- multi-modal test: `loss 0.3200, acc 0.8735`
- multi-modal subtest: `loss 0.2852, acc 0.8921`
- single-modal test: `loss 0.3118, acc 0.8635`
- single-modal subtest: `loss 0.3056, acc 0.8669`

## F1 (positive = sarcastic)

| Model | Test word | Test word+emoji | Subtest word | Subtest word+emoji |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Decision tree | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| Random forest | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| Gradient boosting | 0.7515 | 0.7528 | 0.8376 | 0.8357 |
| BiLSTM + attention | 0.8656 | **0.8686** | 0.8940 | **0.9107** |

## Precision and recall (sklearn only, from the metrics dump)

Test / subtest × word / word+emoji, in the same four-slot order as the JSON lists:

| Model | Recall | Precision |
| --- | --- | --- |
| SVM | 0.783, 0.777, 0.872, 0.826 | 0.762, 0.756, 0.833, 0.882 |
| Decision tree | 0.846, 0.841, 0.907, 0.895 | 0.683, 0.688, 0.772, 0.802 |
| Random forest | 0.864, 0.861, 0.907, 0.907 | 0.786, 0.793, 0.804, 0.862 |

Trees recall sarcasm more aggressively than SVM and pay for it in precision. Random forest is the strongest pooled-vector model; it is still well behind the recurrent stack on test accuracy (0.818 vs 0.874).

## What the deltas say

**Emoji on the full test set.**

| Model | Δ acc (emoji − word) | Δ F1 |
| --- | ---: | ---: |
| SVM | −0.0060 | −0.0059 |
| Decision tree | +0.0030 | +0.0009 |
| Random forest | +0.0035 | +0.0023 |
| Gradient boosting | +0.0015 | +0.0013 |
| BiLSTM + attention | +0.0100 | +0.0030 |

A one-point accuracy bump on 2,000 tweets is 20 extra correct items. That is real for the neural model and in the noise for most trees. SVM gets slightly worse: concatenating a mostly-zero 200-d emoji half is a reasonable way to dilute a margin classifier.

**Emoji on the subtest.**

| Model | Δ acc | Δ F1 |
| --- | ---: | ---: |
| SVM | +0.0108 | +0.0006 |
| Decision tree | +0.0216 | +0.0119 |
| Random forest | +0.0468 | +0.0314 |
| Gradient boosting | 0.0000 | −0.0019 |
| BiLSTM + attention | +0.0252 | +0.0167 |

This is the comparison the project is actually about. When the tweet usually has an emoji, emoji2vec is not a sparse missing-modality problem. Random forest gains the most in absolute accuracy; the neural model starts higher and still gains.

**Order vs pooling.** Hold modality fixed and compare BiLSTM+Attn to random forest on test word-only: 0.8635 vs 0.8145. That 5-point gap is larger than any emoji delta on the same split. Sarcasm in this corpus is often a polarity flip across the sentence (`love` … `#not`, `feeling like a million bucks` … `😅`), which mean-pooling smears.

## Confounders you should report if you cite these numbers

1. **Cue hashtags.** 595 / 2,000 test tweets contain `#not` / `#sarcasm` / … and all 595 are labeled sarcastic. A look-at-hashtags baseline is strong. The neural model can exploit `#not` if it is in GloVe; the examples' toy cue classifier exists to make that ceiling visible.
2. **Subtest class prior.** 61.9% sarcastic vs 50% on test. Accuracy is not comparable across splits without that note.
3. **Shuffle in `ml_read_data`.** Sklearn features are permuted; reported scores are still valid because labels were permuted with them.
4. **Missing pickles.** This clone has DT and GBT pickles only. SVM / RF numbers come from the notebook that loaded files which are not in git. You cannot re-score those two from the repo alone.
5. **No confidence intervals.** One seed, one split, one saved net. Treat the third decimal as decorative.
6. **Copy-paste fallbacks** in `baseline_models.ipynb` (SVC constructed under RF/GBT headings) mean a clean-room retrain from that notebook would not match the 2023 pickles. See [architecture.md](architecture.md).

## What would be a fair follow-up (not done here)

- Ablate cue hashtags (strip `#not` / `#sarcasm` before vectorizing).
- Report emoji-present vs emoji-absent slices of *test* instead of a separate `subtest` file.
- Bootstrap CIs on the 2,000-row test set.
- Replace mean-pool with a small learned projection, still without GloVe, for a modern personal baseline.

`examples/06_toy_baseline.py` is the honest cheap baseline: hashtag cues + emoji presence, no pretrained vectors. It is not a competitor to the 2023 BiLSTM; it is a documented floor / leak check.
