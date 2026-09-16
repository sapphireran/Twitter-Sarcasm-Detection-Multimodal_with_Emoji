# Results

All numbers below are copied from the committed notebook outputs
(`baseline_models.ipynb`, `evaluate_loaded_dl_models.ipynb`,
`get_metrics_of_models.ipynb`). They are historical. Re-running the
notebooks without the original GloVe file and pickles will not
reproduce them bit-for-bit.

Column keys:

- **W** — word embeddings only
- **WE** — word + emoji embeddings
- **test** — 2,000-tweet balanced split
- **sub** — 278-tweet emoji-heavy split

## Accuracy

| Model | test W | test WE | sub W | sub WE |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7265 | 0.7295 | 0.7770 | 0.7986 |
| SVM | 0.7690 | 0.7630 | 0.8129 | 0.8237 |
| Gradient Boosting | 0.7460 | 0.7475 | 0.7950 | 0.7950 |
| Random Forest | 0.8145 | 0.8180 | 0.8058 | 0.8525 |
| Bi-LSTM + attention | **0.8635** | **0.8735** | **0.8669** | **0.8921** |

GBT accuracies come from `baseline_models.ipynb`. The other rows come
from `get_metrics_of_models.ipynb` (which also re-prints the deep
`evaluate()` numbers).

## F1 (positive = sarcastic)

| Model | test W | test WE | sub W | sub WE |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7557 | 0.7566 | 0.8342 | 0.8462 |
| SVM | 0.7722 | 0.7663 | 0.8523 | 0.8529 |
| Random Forest | 0.8232 | 0.8255 | 0.8525 | 0.8839 |
| Bi-LSTM + attention | **0.8656** | **0.8686** | **0.8940** | **0.9107** |

GBT F1 was not stored in the metrics notebook.

## Recall and precision (where recorded)

`get_metrics_of_models.ipynb` printed recall and precision for SVM,
Decision Tree, Random Forest, and the deep model.

### Recall

| Model | test W | test WE | sub W | sub WE |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.846 | 0.841 | 0.907 | 0.895 |
| SVM | 0.783 | 0.777 | 0.872 | 0.826 |
| Random Forest | 0.864 | 0.861 | 0.907 | 0.907 |
| Bi-LSTM + attention | 0.879 | 0.836 | 0.907 | 0.890 |

### Precision

| Model | test W | test WE | sub W | sub WE |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.6828 | 0.6877 | 0.7723 | 0.8021 |
| SVM | 0.7617 | 0.7558 | 0.8333 | 0.8820 |
| Random Forest | 0.7862 | 0.7928 | 0.8041 | 0.8619 |
| Bi-LSTM + attention | 0.8526 | 0.9038 | 0.8814 | 0.9329 |

## What the numbers say

1. **Architecture beats the emoji channel on the full test set.**
   Going from Random Forest W (0.8145) to Bi-LSTM W (0.8635) is a
   larger jump than adding emoji to either model.

2. **Emoji help more on subtest.** WE−W accuracy deltas:

   | Model | Δ test | Δ sub |
   | --- | ---: | ---: |
   | Decision Tree | +0.0030 | +0.0216 |
   | SVM | −0.0060 | +0.0108 |
   | Gradient Boosting | +0.0015 | 0.0000 |
   | Random Forest | +0.0035 | +0.0468 |
   | Bi-LSTM + attention | +0.0100 | +0.0252 |

   Random Forest is the clearest classical winner: almost no full-test
   movement, a 4.7-point subtest jump. The deep model gains a clean
   point on the balanced test set and 2.5 points on subtest.

3. **SVM is the exception.** WE *hurts* full-test accuracy and F1.
   A default RBF `SVC` on a 400-d concat of two averaged spaces is a
   plausible place for the extra 200 zeros / weakly aligned emoji
   dimensions to add noise.

4. **The deep WE model is more precise, not more complete.** On
   full test, recall drops (0.879 → 0.836) while precision rises
   (0.853 → 0.904). The emoji channel appears to suppress some false
   positives — tweets that look sarcastic from wording alone but are
   sincere once the emoji are read as supportive rather than ironic.

5. **Subtest is easier than test for every model.** Positive rate
   is 61.9%, every tweet has non-ASCII, and 129 / 172 sarcastic tweets
   carry `#not` / `#sarcasm` / `#sarcastic(tweet)`. Do not cite
   subtest 0.89 accuracy as the project's headline number. Cite
   **test WE 0.8735 / F1 0.8686** for the deep model.

## Hashtag-only ceiling (this checkout)

`examples/heuristic_baseline.py` scores a transparent rule on the
same CSVs:

- predict sarcastic if the tweet has `#not`, `#sarcasm`,
  `#sarcastic`, or `#sarcastictweet` as a hashtag token
- otherwise predict not sarcastic

That rule is **not** a published baseline from 2023. It exists so
you can see how much of test/subtest is "the tag is in the string".
On test it catches 594 / 1,000 positives and zero false positives
from those tags, so precision is 1.0 and recall is 0.594 if you ignore
the compositional `#not ready` problem (which the regex treats as a
tag when `#not` is a separate token). The deep model still wins on
F1 because it recovers untagged sarcasm.

## Plot

The metrics notebook drew grouped bars (accuracy) plus overlaid
lines (F1) for DT / SVM / RF / Bi-LSTM across the four
(W/WE × test/sub) settings. That figure is embedded in
`get_metrics_of_models.ipynb` as a PNG output; it was not exported as
a standalone file. `examples/report_metrics.py --markdown` reprints
the tables for READMEs.

## Honest limitations

- No error bars, no cross-validation, no seed.
- Classical test features are shuffled inside `ml_read_data`; the
  printed scores assume the loaded pickles were applied to that
  shuffled test matrix in the same session.
- Saved Keras directories are missing `variables/`.
- SVM / RF pickles are missing from `baseline_models/`.
- Train `#not` is noisy (87 tagged tweets labeled 0).
- "Multi-modal" is two text embedding tables, not vision.
