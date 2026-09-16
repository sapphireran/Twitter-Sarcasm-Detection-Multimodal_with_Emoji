# Results

All neural and sklearn figures below are **transcribed from executed 2023 notebook outputs** (`baseline_models.ipynb`, `get_metrics_of_models.ipynb`, `evaluate_loaded_dl_models.ipynb`). They were not re-trained in this documentation pass. The lexical baseline *was* re-run from `examples/lexical_baseline.py` on the checked-in CSVs (NumPy only).

## Headline

- Best model: **BiLSTM + attention with GloVe + emoji2vec** — 87.35% test acc, 89.21% subtest acc, 0.869 / 0.911 F1.
- Emoji fusion is a **small win on the full test set** (+1.0 acc vs word-only BiLSTM) and a **clearer win on the emoji slice** (+2.5 acc, +1.7 F1).
- Among sklearn models, **Random Forest** is the only one that both (a) beats SVM and (b) uses the 400-d concat productively on subtest (+4.7 acc).
- SVM is slightly **hurt** by emoji concat on the full test set (−0.6 acc) and slightly **helped** on subtest (+1.1 acc). Mean-pool concat is not free: it doubles dimension and adds a sparse block.

## Full test set (n = 2,000, balanced)

| Model | Word acc | Word+emoji acc | Δ acc | Word F1 | Word+emoji F1 | Δ F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7265 | 0.7295 | +0.003 | 0.7557 | 0.7566 | +0.001 |
| Gradient Boosting | 0.7460 | 0.7475 | +0.002 | 0.7515 | 0.7528 | +0.001 |
| SVM | 0.7690 | 0.7630 | −0.006 | 0.7722 | 0.7663 | −0.006 |
| Random Forest | 0.8145 | 0.8180 | +0.004 | 0.8232 | 0.8255 | +0.002 |
| BiLSTM + Attention | 0.8635 | **0.8735** | +0.010 | 0.8656 | **0.8686** | +0.003 |

Deep models sit ~5 points of accuracy above the best tree ensemble. That gap is the value of **order**: “I love” + later `#not` is a different object from the mean of `{I, love, #not}`.

## Subtest (n = 278, emoji-bearing, 61.9% positive)

| Model | Word acc | Word+emoji acc | Δ acc | Word F1 | Word+emoji F1 | Δ F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 0.7770 | 0.7986 | +0.022 | 0.8342 | 0.8462 | +0.012 |
| Gradient Boosting | 0.7950 | 0.7950 | 0.000 | 0.8376 | 0.8357 | −0.002 |
| SVM | 0.8129 | 0.8237 | +0.011 | 0.8523 | 0.8529 | +0.001 |
| Random Forest | 0.8058 | **0.8525** | +0.047 | 0.8525 | **0.8839** | +0.031 |
| BiLSTM + Attention | 0.8669 | **0.8921** | +0.025 | 0.8940 | **0.9107** | +0.017 |

Read this table as: *when the tweet actually contains an emoji, giving the model a dedicated emoji table helps.* When most tweets have an all-zero emoji half (full test), concat mostly adds noise for margin-based SVM.

Subtest is **not** an independent sample. It is a filter on test. Confidence intervals on a 278-row slice are wide; treat +4.7 RF acc as directional, not as a precise effect size.

## Loss on the deep models

From `evaluate_loaded_dl_models.ipynb`:

| Checkpoint | Test loss / acc | Subtest loss / acc |
| --- | --- | --- |
| `model/best_model_single_modal` | 0.3118 / 0.8635 | 0.3056 / 0.8669 |
| `model/best_model_multi_modal` | 0.3200 / 0.8735 | 0.2852 / 0.8921 |

Multi-modal test **loss is slightly worse** than single-modal (0.320 vs 0.312) while **accuracy is better**. That usually means a few confident mistakes versus more threshold-correct calls — worth a calibration plot in a rerun, not something the 2023 notebooks plotted.

## Lexical floor (recomputed here)

`examples/lexical_baseline.py` trains L2 logistic regression on hand-built cues (`#not`, `#sarcasm`, elongated words, contrast templates, emoji faces, punctuation). Recomputed on the checked-in CSVs (see `docs/generated/lexical_baseline.md`):

| Split | Acc | Prec | Rec | F1 |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.665 | 0.688 | 0.511 | 0.587 |
| Test | 0.641 | 0.636 | 0.657 | 0.646 |
| Subtest | 0.788 | 0.827 | 0.831 | 0.829 |

`#not` is the heaviest standardized weight (+1.18). Mentions push *against* sarcasm (−0.48) because `<user>` lives only on (mostly sincere) train rows and is absent at test — the cue is a split artifact, not a linguistic one.

64% test acc is the spelling floor. The multi-modal BiLSTM at 87% still has ~23 points of room that order + distributed features have to explain. On subtest the floor jumps to 79% because `#not` is denser there (110/172 sarcastic rows); the BiLSTM’s 89% is a smaller but real gap.

A model that never sees embeddings but sees `#not` already looks competent on this dump — which is why the neural gains on the **emoji slice beyond hashtags** are the claim worth defending.

## What not to claim

- **Not** “we beat SOTA sarcasm detection.” No SemEval / iSarcasm official split is used here.
- **Not** “emoji2vec always helps.” SVM test acc went down.
- **Not** “subtest proves generalization to unseen emoji.” Train tokenizer + frozen tables + subtest⊂test say otherwise.
- **Not** a reproducible SavedModel reload. The `model/*/variables` shards are missing; the notebook numbers assume a complete 2023 checkpoint that is only partially uploaded.

## Plot

`get_metrics_of_models.ipynb` ends with a grouped bar chart of accuracies and F1 scores. Re-create it from the tables above if the notebook kernel is not available; the underlying scalars are complete in the cell stdout.
