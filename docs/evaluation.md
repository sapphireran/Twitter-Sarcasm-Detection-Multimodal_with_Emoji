# Evaluation

All headline numbers below are copied from the executed cells in `get_metrics_of_models.ipynb` and `evaluate_loaded_dl_models.ipynb` (June 2023). They are **not** recomputed in this documentation pass: GloVe and several sklearn pickles are missing from the snapshot, and the Keras checkpoints need a matching TF / custom-object load.

`examples/reprint_course_results.py` prints the same table. `examples/cue_baseline.py` *does* recompute a hashtag / clash heuristic on the current CSVs so you have one live number.

## Protocol

| Item | What the notebooks did |
| --- | --- |
| Labels | `1` = sarcastic, `0` = not. Positive-class precision / recall / F1. |
| Main hold-out | `dataset/test_*.csv` (2,000 rows, balanced). |
| Emoji slice | `dataset/subtest_*.csv` (278 rows, 99.6% contain an emoji). |
| Deep features | Train tokenizer + train `maxlen` (78), `preprocess_test` for both hold-outs. |
| sklearn features | `ml_read_data` mean-pool; unseeded shuffle does not affect metrics once predictions are made. |
| Deep metric source | `model.evaluate` for accuracy; `predict` + sklearn for F1 / P / R. |

Four columns appear everywhere:

1. Single-modal, `test`
2. Multi-modal, `test`
3. Single-modal, `subtest`
4. Multi-modal, `subtest`

## Results (2023 notebooks)

Accuracy / F1 / recall / precision. GBT only has accuracy and F1 in the executed cells.

### `test` (2,000 tweets)

| Model | Modal | Acc | F1 | Rec | Prec |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | text | 0.769 | 0.772 | 0.783 | 0.762 |
| SVM | +emoji | 0.763 | 0.766 | 0.777 | 0.756 |
| Decision tree | text | 0.726 | 0.756 | 0.846 | 0.683 |
| Decision tree | +emoji | 0.730 | 0.757 | 0.841 | 0.688 |
| Random forest | text | 0.814 | 0.823 | 0.864 | 0.786 |
| Random forest | +emoji | 0.818 | 0.826 | 0.861 | 0.793 |
| Gradient boosting | text | 0.746 | 0.751 | — | — |
| Gradient boosting | +emoji | 0.748 | 0.753 | — | — |
| Bi-LSTM + Att | text | 0.8635 | 0.866 | 0.879 | 0.853 |
| Bi-LSTM + Att | +emoji | **0.8735** | **0.869** | 0.836 | **0.904** |

### `subtest` (278 emoji-heavy tweets)

| Model | Modal | Acc | F1 | Rec | Prec |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | text | 0.813 | 0.852 | 0.872 | 0.833 |
| SVM | +emoji | 0.824 | 0.853 | 0.826 | 0.882 |
| Decision tree | text | 0.777 | 0.834 | 0.907 | 0.772 |
| Decision tree | +emoji | 0.799 | 0.846 | 0.895 | 0.802 |
| Random forest | text | 0.806 | 0.852 | 0.907 | 0.804 |
| Random forest | +emoji | 0.853 | 0.884 | 0.907 | 0.862 |
| Gradient boosting | text | 0.795 | 0.838 | — | — |
| Gradient boosting | +emoji | 0.795 | 0.836 | — | — |
| Bi-LSTM + Att | text | 0.867 | 0.894 | 0.907 | 0.881 |
| Bi-LSTM + Att | +emoji | **0.892** | **0.911** | 0.890 | **0.933** |

`evaluate_loaded_dl_models.ipynb` independently reports the same Bi-LSTM accuracies: multi-modal `0.8735` / `0.8921`, single-modal `0.8635` / `0.8669`.

## What the table is saying

1. **Sequence models beat mean-pooling.** Even the single-modal Bi-LSTM (0.8635 test acc) is 5 points above the best sklearn model (RF 0.8145). Order and the attention pool are doing work that a 200-d mean cannot.

2. **Emoji vectors help most where emoji are actually present.** On mixed `test`, multi-modal RF gains +0.4 acc and the Bi-LSTM gains +1.0. On `subtest`, RF gains **+4.7** and the Bi-LSTM gains **+2.5**. SVM’s test score slightly *drops* with emoji (+400-d concat on a kernel SVM is not free); its subtest score still rises.

3. **The multi-modal Bi-LSTM is a precision model on `test`.** Recall falls 0.879 → 0.836 while precision rises 0.853 → 0.904. Adding emoji2vec makes the network more conservative about calling a tweet sarcastic unless the emoji evidence agrees. That is the behaviour you want if the product cost of a false positive is high.

4. **Hashtag leakage is the silent ceiling.** `examples/explore_dataset.py` shows that *every* sarcasm cue tag on `test` and `subtest` sits on a positive label (613 / 1,000 sarcastic test tweets). A cue-only rule is therefore an unfair competitor on these splits. Run `examples/cue_baseline.py` and compare: if your new neural net cannot beat a `#not` detector on `test`, it is not beating the annotation process.

5. **Decision-tree multi-modal numbers are slightly suspect.** See the `SVC()` typo noted in [training.md](training.md). The DT +emoji row may be an SVM. RF and Bi-LSTM do not have that bug.

## Live cue baseline (recomputed from the CSVs)

`examples/cue_baseline.py` implements three rules that need no embeddings:

| Rule | Idea |
| --- | --- |
| `cue-tag` | Predict sarcastic iff a distant-supervision hashtag is present (`#not`, `#sarcasm`, `#yeahright`, …). |
| `clash` | Predict sarcastic iff a positive cue word (`love`, `great`, `yay`, …) co-occurs with a negative cue (`hate`, `dirty`, `3am`, 😒, …) or a cue tag. |
| `majority` | Always predict the training majority (`0`). |

Reproduce:

```bash
python examples/cue_baseline.py
```

Live numbers from the current CSVs (recomputed in this repo, not from 2023):

| Split | Rule | Acc | F1 | Rec | Prec |
| --- | --- | ---: | ---: | ---: | ---: |
| `test` | majority (`0`) | 0.500 | 0.000 | 0.000 | 0.000 |
| `test` | cue-tag | 0.806 | 0.760 | 0.613 | **1.000** |
| `test` | clash | 0.812 | 0.772 | 0.637 | 0.980 |
| `subtest` | cue-tag | 0.863 | 0.876 | 0.779 | **1.000** |
| `subtest` | clash | 0.888 | 0.903 | 0.843 | 0.973 |

Cue-tag already matches Random Forest *accuracy* on `test` (0.806 vs 0.814) because a third of that split is an explicit sarcasm hashtag on a positive label. The Bi-LSTM still wins (0.874 / 0.892) by catching untagged clashes such as `useless` vs `great`. Interpret the heuristic as a *cheat-code floor*, not as a model in the 2023 table.

## How to reload the deep models

```python
import tensorflow as tf
from attention_layer import Attention

model = tf.keras.models.load_model(
    "model/best_model_multi_modal",
    custom_objects={"Attention": Attention},
)
```

The 2023 evaluation notebook called `load_model` without `custom_objects` and it worked on that TF build (the SavedModel already bundled the layer). A current TensorFlow will usually require the `custom_objects` map. You also need the train tokenizer and `maxlen=78` to build `X_test`; those are **not** serialized next to the SavedModel. Rebuild them with `ReadOpen` + `Preprocess` on `dataset/train_*.csv` and the same GloVe / emoji2vec files.

The snapshot under `model/best_model_*` contains `saved_model.pb` and `keras_metadata.pb` only. Some TF versions also expect a `variables/` directory. If `load_model` errors on missing variables, the weight blobs were not uploaded in `d60afa1` and you cannot evaluate the checkpoints from this clone alone.

## Reporting checklist for a retrain

When you replace a number in this page, record all four of:

- split (`test` / `subtest`)
- modality (text / +emoji)
- metric (acc / F1 / P / R, positive class)
- seed / pickle / checkpoint hash

Do not average `test` and `subtest`. They answer different questions.
