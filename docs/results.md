# Results

All numbers below are transcribed from executed cells in `get_metrics_of_models.ipynb` (June 2023) and cross-checked against `baseline_models.ipynb` and `evaluate_loaded_dl_models.ipynb` where those notebooks print the same metric.

Column key:

- **single** = GloVe averages only, or the mixed table with `get_emoji2vec=False`
- **multi** = GloVe ∥ emoji2vec (baselines) or mixed table with emoji2vec fill
- **test** = 2,000 balanced tweets
- **subtest** = 278 emoji / marker-heavy tweets

## Accuracy

| Model | test single | test multi | Δ test | subtest single | subtest multi | Δ subtest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.7690 | 0.7630 | −0.0060 | 0.8129 | 0.8237 | +0.0108 |
| Decision tree | 0.7265 | 0.7295 | +0.0030 | 0.7770 | 0.7986 | +0.0216 |
| Random forest | 0.8145 | 0.8180 | +0.0035 | 0.8058 | 0.8525 | +0.0468 |
| Gradient boosting | 0.7460 | 0.7475 | +0.0015 | 0.7950 | 0.7950 | 0.0000 |
| Bi-LSTM + attention | 0.8635 | **0.8735** | +0.0100 | 0.8669 | **0.8921** | +0.0252 |

## F1 (positive = sarcastic)

| Model | test single | test multi | Δ test | subtest single | subtest multi | Δ subtest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.7722 | 0.7663 | −0.0059 | 0.8523 | 0.8529 | +0.0006 |
| Decision tree | 0.7557 | 0.7566 | +0.0009 | 0.8342 | 0.8462 | +0.0119 |
| Random forest | 0.8232 | 0.8255 | +0.0023 | 0.8525 | 0.8839 | +0.0314 |
| Gradient boosting | 0.7515 | 0.7528 | +0.0013 | 0.8376 | 0.8357 | −0.0019 |
| Bi-LSTM + attention | 0.8656 | **0.8686** | +0.0030 | 0.8940 | **0.9107** | +0.0167 |

## Precision and recall on the main test set

From the `*_list` dumps at the bottom of the metrics notebook. Order inside each 4-tuple is `[test single, test multi, subtest single, subtest multi]`.

| Model | test P single | test P multi | test R single | test R multi |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.762 | 0.756 | 0.783 | 0.777 |
| Decision tree | 0.683 | 0.688 | 0.846 | 0.841 |
| Random forest | 0.786 | 0.793 | 0.864 | 0.861 |
| Bi-LSTM + attention | 0.853 | **0.904** | 0.879 | 0.836 |

Subtest precision / recall:

| Model | sub P single | sub P multi | sub R single | sub R multi |
| --- | ---: | ---: | ---: | ---: |
| SVM | 0.833 | 0.882 | 0.872 | 0.826 |
| Decision tree | 0.772 | 0.802 | 0.907 | 0.895 |
| Random forest | 0.804 | 0.862 | 0.907 | 0.907 |
| Bi-LSTM + attention | 0.881 | **0.933** | 0.907 | 0.890 |

Gradient boosting precision / recall were printed as F1 / accuracy only; they are not in the `*_list` dump.

## How to read the lift

1. **Architecture beats fusion on the balanced test set.** Going from RF (0.818) to Bi-LSTM (0.874) is a much larger jump than adding emoji to either model.
2. **Fusion is not free.** SVM loses 0.6 acc on test when the emoji half is concatenated. GBT is flat on the subtest. A second channel that is often zeros is extra variance.
3. **Fusion shows up where emoji actually occur.** RF +4.7 acc and Bi-LSTM +2.5 acc on the subtest. That is the result the project is about.
4. **The multimodal LSTM is a precision model on test.** Recall drops (0.879 → 0.836) while precision jumps (0.853 → 0.904). It says "sarcastic" less often and is more often right when it does. On the subtest both precision and F1 go up; recall stays high.
5. **Decision trees leak sarcasm.** High recall, low precision, weakest accuracy. They are in the table as a weak baseline, not as a contender.

## Deep-model `evaluate` traces

From `evaluate_loaded_dl_models.ipynb`:

```
multi-modal  test:    loss 0.3200  acc 0.8735   (63 steps)
multi-modal  subtest: loss 0.2852  acc 0.8921   (9 steps)
single-modal test:    loss 0.3118  acc 0.8635
single-modal subtest: loss 0.3056  acc 0.8669
```

The metrics notebook loads the same graphs under longer folder names that encode those scores (`best_model_we_0.8734...`). Those folders are not in git; `model/best_model_*` is the surviving copy.

## Plots

The metrics notebook draws a grouped bar chart of accuracy and F1 for SVM, DT, RF, and Bi-LSTM (GBT is omitted from that figure). The PNG is stored inline in the `.ipynb` and is not exported to `docs/`.

## What would change the table

- Seeding `ml_read_data`. The classical features are shuffled with an unseeded permutation. Re-fitting sklearn models will not bit-match these rows.
- A real RF / GBT / DT fallback train. The notebook's `except` branches fit the wrong estimator (see [known-issues.md](known-issues.md)).
- Newer TensorFlow. The SavedModels use `module_wrapper_*` layers from TF 2.x wrapping Keras 2. They may refuse to load.
- A different subtest definition. The 278-row file is a curated slice; resampling test would not reproduce those columns.

The `examples/` toy classifier is **not** in this table. It exists to show that fusion can separate a 16-tweet corpus, not to compete with the 2023 numbers.
