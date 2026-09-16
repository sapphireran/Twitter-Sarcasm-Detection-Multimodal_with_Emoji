# Experiment log — June 2023

Personal notes from the UCPH CCS2 sarcasm project. Dates come from the executed cells in `get_metrics_of_models.ipynb` (5–6 June 2023). I am writing this later from those outputs plus a recount of the CSVs, not from a contemporaneous lab book.

Notation I used then and still use here:

- **W** — words only. GloVe Twitter 200-d. Emoji tokens that GloVe does not know stay zero (deep model) or are ignored in the mean (baselines).
- **WE** — words + emoji. Same GloVe table, plus emoji2vec for emoji tokens / a concatenated 400-d mean for sklearn.
- **full test** — `dataset/test_*.csv`, 2,000 rows, 1,000 / 1,000.
- **subtest** — `dataset/subtest_*.csv`, 278 rows, almost all emoji-bearing.

I did not do a proper nested CV. Test was used as the validation mirror in `evaluate_loaded_dl_models.ipynb` (`X_val = X_test`). Treat the deep-model numbers as "best snapshot on the set I was watching," not as a locked holdout.

## Why I ran these particular comparisons

Sarcasm on Twitter is often a mismatch: positive words, exhausted face; a compliment, a skull. I wanted to know whether a cheap second embedding for emoji moved anything, or whether sequence models already absorb the signal from the Unicode characters as unknown tokens.

The experimental skeleton stayed fixed:

1. Same tokenizer (`TweetTokenizer`, lowercased).
2. Same GloVe file.
3. Same emoji2vec file (`emoji2vec_twitter.bin`, not the older `emoji2vec.bin` sitting in the repo root).
4. Same four sklearn defaults and one Keras stack.
5. Report accuracy and F1 on both evaluation slices.

I did not sweep kernels, tree depths, or LSTM widths in a way I recorded. Defaults plus the architecture in `dl_model.py`.

## Data snapshot I actually trained on

Recounted from the CSVs in this checkout:

| Split | n | sarcastic | emoji tweets | mean whitespace tokens |
| --- | ---: | ---: | ---: | ---: |
| train | 39,780 | 18,488 (46.5%) | 5,470 (13.8%) | 16.46 |
| test | 2,000 | 1,000 (50.0%) | 277 (13.9%) | 16.51 |
| subtest | 278 | 172 (61.9%) | 277 (99.6%) | 17.70 |

Train and test have almost the same emoji rate. Subtest is a different distribution: one row without emoji, sarcastic-heavy. Any "emoji helps +2.5 acc on subtest" claim has to live next to "that slice is 99.6% emoji."

On train, emoji is slightly *more* common in non-sarcastic tweets (3,193 / 21,292) than sarcastic ones (2,277 / 18,488). So "has emoji" is not a sarcasm detector by itself. The useful part, if any, is *which* emoji sits next to *which* words.

## Classical models

Trained with `data_utils.ml_read_data`. That function:

1. Tokenizes.
2. Mean-pools GloVe over in-vocabulary tokens → 200-d `X`.
3. Mean-pools emoji2vec the same way, concatenates → 400-d `X_emoji`.
4. Shuffles train with `np.random.permutation` and applies the same index order to both views.

If a tweet has no GloVe tokens, the W vector is zeros. If it has no emoji2vec tokens, the emoji half is zeros, so WE is `[glove; 0]`.

Sklearn objects were default constructors. I did not grid-search C, `max_depth`, or `n_estimators`.

### Full test

| Model | Acc W | Acc WE | F1 W | F1 WE | Rec W | Rec WE | Prec W | Prec WE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.769 | 0.763 | 0.772 | 0.766 | 0.783 | 0.777 | 0.762 | 0.756 |
| Decision tree | 0.7265 | 0.7295 | 0.756 | 0.757 | 0.846 | 0.841 | 0.683 | 0.688 |
| Random forest | 0.8145 | 0.818 | 0.823 | 0.826 | 0.864 | 0.861 | 0.786 | 0.793 |
| Gradient boosting | 0.746 | 0.7475 | 0.751 | 0.753 | — | — | — | — |

GBT precision/recall were computed in the notebook aggregation path but I did not keep a clean printed block the way I did for SVM / DT / RF. Acc and F1 above are from the dedicated GBT cells.

### Subtest

| Model | Acc W | Acc WE | F1 W | F1 WE | Rec W | Rec WE | Prec W | Prec WE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.813 | 0.824 | 0.852 | 0.853 | 0.872 | 0.826 | 0.833 | 0.882 |
| Decision tree | 0.777 | 0.799 | 0.834 | 0.846 | 0.907 | 0.895 | 0.772 | 0.802 |
| Random forest | 0.806 | 0.853 | 0.852 | 0.884 | 0.907 | 0.907 | 0.804 | 0.862 |
| Gradient boosting | 0.795 | 0.795 | 0.838 | 0.836 | — | — | — | — |

### What I take from the baselines

- Mean-pooled GloVe is a surprisingly strong bag-of-words for this label. Forest at 81.5% is the number I would quote if someone asked "do you even need a recurrent net?"
- Concatenating emoji2vec is not free accuracy. SVM lost 0.6 acc on the full test. Boosting barely moved. The forest picked up 0.35 acc on full test and a much clearer **+4.7 acc / +3.2 F1** on the emoji slice.
- Decision trees love recall and leak precision (0.85 rec / 0.68 prec on full test). They call sarcasm too often. Forest is the same direction, milder.
- SVM on subtest is the interesting precision/recall swap: WE recall dropped (0.872 → 0.826) while precision jumped (0.833 → 0.882). F1 stayed flat. The extra channel made it fussier, not uniformly better.

### Copy-paste I should not pretend did not happen

In `baseline_models.ipynb` the "train if pickle missing" cells are wrong in three places:

- Decision-tree **WE** branch constructs `SVC()`.
- Random-forest **W** branch constructs `SVC()`.
- Gradient-boosting **W** branch constructs `SVC()`.

The printed scores in that notebook match `get_metrics_of_models.ipynb`, and those loads used pickles named `dt_classifier.pkl`, `rf_…`, `gbt_…`. I believe the numbers come from earlier, correctly named fits, not from re-running the broken except-blocks. This checkout only still has the DT and GBT pickles. SVM and RF pickles are gone.

If I ever retrain from this notebook as written, those three cells will silently emit SVMs. Do not do that and then compare to the 2023 table.

## Deep model

Architecture is `PrepModel` in `dl_model.py`. Both saved graphs report 2,510,848 parameters. Pad length in the summaries is 78.

Embedding construction is the actual experiment:

```text
for each vocab token:
    if token in GloVe:
        use GloVe
    else:
        peel emoji codepoints out of the token
        if get_emoji2vec and those codepoints are in emoji2vec:
            mean-pool them into the same 200-d slot
        else:
            zeros
```

So WE is not a second tower. It is a less empty embedding table. The BiLSTMs see the same width either way.

Saved snapshot names from the metrics notebook:

```text
model/best_model_w_0.8634999990463257_sub_0.866906464099884
model/best_model_we_0.8734999895095825_sub_0.8920863270759583
```

Those directories are not in git. The current `model/best_model_{single,multi}_modal` folders are the same graphs without the `variables/` weights.

### Numbers I recorded

From `model.evaluate` and `f1_score` on thresholded `predict > 0.5`:

| Slice | Acc W | Acc WE | F1 W | F1 WE | Rec W | Rec WE | Prec W | Prec WE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full test | 0.8635 | 0.8735 | 0.8656 | 0.8686 | 0.879 | 0.836 | 0.853 | 0.904 |
| subtest | 0.8669 | 0.8921 | 0.8940 | 0.9107 | 0.907 | 0.890 | 0.881 | 0.933 |

Loss on full test was 0.3118 (W) vs 0.3200 (WE). WE is more accurate and slightly higher loss. I did not chase that. Possible explanations I still like: sharper calibration on the sarcastic class (precision 0.90) with more confident mistakes on the rest, or just a different epoch.

### How I read the deep-model delta

On the balanced test set the emoji table is a **+1.0 acc, +0.3 F1** bump. Small, but it showed up in both acc and F1, and the same direction as the forest.

The more interesting movement is precision/recall:

- W: high recall (0.879), solid precision (0.853).
- WE: recall drops to 0.836, precision jumps to 0.904.

Same story as SVM on the subtest. Emoji evidence makes the net less willing to call sarcasm unless the face matches the words. On a balanced test set that trade is a net win. On a sarcastic-heavy subtest, F1 still rises because precision was the slack variable.

Subtest **+2.5 acc / +1.7 F1** is the number I would put on a poster if I had to pick one, with the caption "evaluated on tweets that actually contain emoji."

## Things I did not run (and still would)

1. **Emoji-only ablation on the full test restricted to the 277 emoji rows.** Subtest is close, but it is also 62% sarcastic. I should have carved the same 277 rows out of `test_*` and scored W vs WE there.
2. **A "has-emoji" binary feature.** Cheap sanity check that the 200-d emoji mean is doing more than flagging presence. Train rates already suggest presence is weakly anti-correlated with sarcasm.
3. **Frozen vs fine-tuned embeddings.** I froze GloVe. Fine-tuning 200-d on 40k tweets might eat the emoji2vec gap or blow it up.
4. **A real validation split.** Using test as val is the ugliest thing in the notebooks.
5. **Error slices.** I never dumped false positives that W got right and WE flipped, or the reverse. That would tell me whether the precision bump is "crying-laughing after a sincere compliment" or something duller.
6. **Hyperparameter search.** One stack, one seed story I did not write down.

## Chronology I can still see

- 5 June 2023, ~22:44 — load GloVe from `glove_tt.txt` (text, not the binary the other notebook wants) and `emoji2vec_twitter.bin`. About two minutes.
- 22:45–22:47 — sklearn feature matrices and baseline predictions.
- 22:47–22:49 — deep-model preprocess (`get_emoji2vec=True` even when evaluating the W snapshot; that only matters for training-time table fill, and here we were loading already-trained graphs).
- 22:49–23:03 — load the two Keras snapshots, evaluate, F1.
- 6 June 2023, 02:13 — later cell re-imports. Plot cells after that.

`baseline_models.ipynb` and `evaluate_loaded_dl_models.ipynb` are the same experiments with fewer metrics and slightly different filenames (`glove.twitter.27B.200d.bin`, `svm_classifier.pkl` vs `svm_model.pkl`).

## Bottom line I would tell myself in 2023

Use the BiLSTM + attention. Quote **87.4% / 86.9 F1** on the balanced test with emoji2vec filled in. Mention that the same model is **89.2% / 91.1 F1** on the emoji slice. Do not claim a large multi-modal miracle: most of the accuracy is already in the word sequence. Emoji is a small, consistent, precision-leaning bonus when the faces are actually there.

If someone only lets me ship a sklearn model, ship the forest, and only bother with the 400-d concat if the traffic looks like the subtest.
