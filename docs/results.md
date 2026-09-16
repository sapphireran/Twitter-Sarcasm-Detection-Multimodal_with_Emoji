# Recorded results (June 2023)

Numbers below are copied from `get_metrics_of_models.ipynb` and
`evaluate_loaded_dl_models.ipynb`. They are historical scores, not a live
re-evaluation. The SavedModel weights are no longer complete in this clone.

Conditions:

- **W / single-modal** — GloVe word vectors only
- **WE / multi-modal** — GloVe + emoji2vec
- **test** — 2,000 official test tweets, balanced
- **subtest** — 278 emoji-bearing test tweets

Regenerate the markdown tables with:

```bash
python3 -m examples.metrics_table --metric accuracy --markdown
python3 -m examples.metrics_table --metric f1 --markdown
```

## Accuracy (%)

| Model | W test | WE test | W subtest | WE subtest |
| --- | ---: | ---: | ---: | ---: |
| SVM | 76.90 | 76.30 | 81.29 | 82.37 |
| Decision Tree | 72.65 | 72.95 | 77.70 | 79.86 |
| Random Forest | 81.45 | 81.80 | 80.58 | 85.25 |
| Gradient Boosting | 74.60 | 74.75 | 79.50 | 79.50 |
| Bi-LSTM + Attention | 86.35 | 87.35 | 86.69 | 89.21 |

## F1 (%)

| Model | W test | WE test | W subtest | WE subtest |
| --- | ---: | ---: | ---: | ---: |
| SVM | 77.22 | 76.63 | 85.23 | 85.29 |
| Decision Tree | 75.57 | 75.66 | 83.42 | 84.62 |
| Random Forest | 82.32 | 82.55 | 85.25 | 88.39 |
| Gradient Boosting | 75.15 | 75.28 | 83.76 | 83.57 |
| Bi-LSTM + Attention | 86.56 | 86.86 | 89.40 | 91.07 |

## Recall and precision for the two strongest models

| Model | Metric | W test | WE test | W subtest | WE subtest |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | recall | 86.40 | 86.10 | 90.70 | 90.70 |
| Random Forest | precision | 78.62 | 79.28 | 80.41 | 86.19 |
| Bi-LSTM + Attention | recall | 87.90 | 83.60 | 90.70 | 88.95 |
| Bi-LSTM + Attention | precision | 85.26 | 90.38 | 88.14 | 93.29 |

## Reading the table

1. **The LSTM wins on every official split.** Frozen GloVe plus two
   bidirectional LSTMs and attention is about five points above random forest
   on test accuracy.
2. **Emoji2vec is a small official-test gain and a clearer subtest gain.**
   Bi-LSTM test accuracy goes 86.35 → 87.35. Subtest goes 86.69 → 89.21. Random
   forest subtest jumps 80.58 → 85.25. That matches the construction of
   subtest: if there is no emoji, the extra 200 dimensions are zeros.
3. **SVM slightly *drops* on official test when emoji are added** (76.90 →
   76.30) and only recovers on subtest. Mean-pooled concatenated vectors are a
   blunt multi-modal fusion.
4. **Precision of the multi-modal LSTM on subtest is 93.29%.** Once emoji are
   present, the false-positive rate falls hard.
5. **Recall of the multi-modal LSTM on official test falls** (87.90 → 83.60)
   while precision rises (85.26 → 90.38). The emoji channel makes the model
   pickier, not just more accurate.

## Architecture size

`evaluate_loaded_dl_models.ipynb` printed the same parameter count for both
saved nets:

```text
Embedding 200-d, length 78
BiLSTM 256  → 512
BiLSTM 256  → 512
Attention   → 512
Dense(1)
Total params: 2,510,848
```

Loss on the official test set was 0.3118 (W) and 0.3200 (WE). The multi-modal
net is not a lower-loss model; it is a better *decision* model on this split.

## What the new lexical baseline is for

`python -m examples.lexical_baseline_demo` trains a 19-feature logistic model
on a random train subset. It is not meant to beat 87%. It is meant to show
that `#not`, contrast, and polarity already move the needle, and that the
official test set is cue-heavier than train. Treat those scores as a classroom
sanity check, not as a 2023 result.
