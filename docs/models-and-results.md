# Models and recorded results

All numbers on this page are **transcribed from executed notebook outputs** in `baseline_models.ipynb` and `get_metrics_of_models.ipynb`. They were not recomputed in this documentation pass. If a pickle or SavedModel fails to load, treat the notebook output as the historical record.

Notation:

- **text** — single-modal (GloVe averages, or a GloVe-only embedding matrix)
- **text+emoji** — multimodal (concatenated averages, or GloVe+emoji2vec matrix)
- **test** — 2,000 balanced tweets
- **subtest** — 278 emoji-heavy tweets

## Accuracy

| Model | test text | test text+emoji | subtest text | subtest text+emoji |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| Gradient boosting | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| SVM | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| Random forest | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| BiLSTM + attention | 0.8635 | 0.8735 | 0.8669 | 0.8921 |

## F1 (sarcastic class)

| Model | test text | test text+emoji | subtest text | subtest text+emoji |
| --- | ---: | ---: | ---: | ---: |
| Decision tree | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| Gradient boosting | 0.7515 | 0.7528 | 0.8376 | 0.8357 |
| SVM | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Random forest | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| BiLSTM + attention | 0.8656 | 0.8686 | 0.8940 | 0.9107 |

## Precision and recall (classical models on the four settings)

Order inside each list: test text, test text+emoji, subtest text, subtest text+emoji.

**SVM**

| Metric | values |
| --- | --- |
| Accuracy | 0.769, 0.763, 0.813, 0.824 |
| F1 | 0.772, 0.766, 0.852, 0.853 |
| Recall | 0.783, 0.777, 0.872, 0.826 |
| Precision | 0.762, 0.756, 0.833, 0.882 |

**Decision tree**

| Metric | values |
| --- | --- |
| Accuracy | 0.727, 0.730, 0.777, 0.799 |
| F1 | 0.756, 0.757, 0.834, 0.846 |
| Recall | 0.846, 0.841, 0.907, 0.895 |
| Precision | 0.683, 0.688, 0.772, 0.802 |

**Random forest**

| Metric | values |
| --- | --- |
| Accuracy | 0.815, 0.818, 0.806, 0.853 |
| F1 | 0.823, 0.826, 0.852, 0.884 |
| Recall | 0.864, 0.861, 0.907, 0.907 |
| Precision | 0.786, 0.793, 0.804, 0.862 |

Decision trees recall sarcastic tweets aggressively and pay for it in precision. Random forest is the only classical model that stays above 0.80 accuracy on both test settings. SVM is linear-ish in the default RBF/kernel setup and does not benefit from the extra 200 emoji dimensions on the balanced test set.

## Deep model extras

From `evaluate_loaded_dl_models.ipynb`, the multimodal checkpoint reported:

```text
test   loss 0.3200   acc 0.8735
subtest loss 0.2852   acc 0.8921
```

The printed `summary()` shows two bidirectional layers (512 units out) and a 78-step embedding. See [`architecture.md`](architecture.md).

## How to read the table

1. **Sequence models beat bag-of-embeddings by a wide margin.** The jump from random forest (0.81–0.82 test acc) to BiLSTM+attention (0.86–0.87) is larger than any emoji increment. Order and a learned pooling head matter more than the extra emoji channel on this data.

2. **Emoji help most where emoji are actually present.** On the balanced test set, multimodal minus single-modal is +0.3 (DT), +0.15 (GBT), −0.6 (SVM), +0.35 (RF), +1.0 (BiLSTM). On the subtest it is +2.2 (DT), 0.0 (GBT), +1.1 (SVM), **+4.7 (RF)**, **+2.5 (BiLSTM)**. The subtest was chosen to contain emoji; that is the split where `emoji2vec` has something to do.

3. **SVM’s small multimodal *drop* on test is not a mystery.** Concatenating a noisy 200-d emoji average onto a 200-d word average doubles the dimension and injects many exact-zero rows (tweets with no in-vocabulary emoji). An untuned `SVC()` can easily waste margin on those coordinates.

4. **Subtest accuracy is not comparable to test accuracy as a ranking of “how good the model is.”** The subtest is smaller, class-imbalanced (62% sarcastic), and enriched for the multimodal feature. Use it as a diagnostic, not as a second test set to average in.

5. **Hashtag leakage sits under every number.** A model that sees `#not` is partly solving a meta-labeling problem. The heuristic example script isolates that effect; the original notebooks did not.

## Heuristic leakage check (added with the examples)

`examples/heuristic_baseline.py` fits a Bernoulli Naive Bayes model on a dozen hand-built features, once **with** `#not` / `#sarcasm` / `#yeahright` as a feature and once with that feature forced off. Same train/test CSVs, no GloVe.

| Setting | test acc | test F1 | subtest acc | subtest F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 0.500 | — | 0.619 | — |
| Bernoulli + leak tags | **0.753** | 0.761 | **0.842** | 0.870 |
| Bernoulli, leak tags removed | 0.557 | 0.620 | 0.576 | 0.617 |

The leak model’s 0.753 test accuracy is already in the same band as the 2023 SVM (0.769) and well above an untuned decision tree (0.727). Turning the hashtag feature off collapses the same classifier to 0.557. On the emoji-heavy subtest the drop is even sharper (0.842 → 0.576).

That does **not** mean the BiLSTM is “just a hashtag detector.” It does mean every published number in the tables above should be read as *including* the distant-supervision tags. The strongest single log-odds feature in the leak model is `has_leak_hashtag` (+3.82); the next, `has_sarcastic_emoji`, is only +0.42.

Re-run:

```bash
python3 examples/heuristic_baseline.py
```

## What was not reported

The notebooks do not include:

- per-tweet error analysis
- attention alignment plots
- McNemar / bootstrap significance tests
- calibration (reliability of the sigmoid)
- an ablation that removes sarcasm hashtags
- learning curves or epoch counts for the LSTM

Those are the first things I would add if this project were restarted, before changing the architecture.
