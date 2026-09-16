# Limitations

## Distant-supervision leakage

A large share of sarcastic labels are recoverable from the same
hashtags that authors used to mark sarcasm. `#not` alone covers 25.8%
of the official test set at 99.6% precision. Any model that can see
raw tokens — including the LSTM embedding table — can spend capacity
on those tags. Slice metrics on `no_explicit_cue` are the honest
number for “did we detect sarcasm.”

`#not ready yet` is the documented false positive: the token matches,
the tweet is often sincere.

## Subtest is not a third split

Quoting subtest next to test as if they were independent overstates
WE. Subtest is a filter on test. Report it as “emoji-bearing test
slice.”

## Incomplete artifacts

Missing GloVe, missing SavedModel variables, and missing SVM / RF
pickles mean the headline 0.8735 / 0.8921 figures are
**notebook outputs**, not reloadable models.

## Notebook bugs that did not affect the published scores

`baseline_models.ipynb` retraining fallbacks construct `SVC()` for
several non-SVM models. `get_metrics_of_models.ipynb` still points at
pickle names (`svm_model.pkl`) that are not in the tree.

`Preprocess` sizes the embedding matrix by **line count**, not vocab
size. Harmless on this train split; wrong as a general pattern.

The attention bias is `(timesteps,)`. Combined with `padding='post'`,
the model can learn “prefer earlier tokens” as a positional prior.

## No company data, no live Twitter API

The CSVs are a 2023 class snapshot. This branch does not call any
social network, does not scrape, and does not add employee or
production code.

## Archive lab is not the 2023 model

The hashed logistic and the hashtag rule are teaching tools. If they
land near SVM on the *cued* slice and collapse on the *uncued* slice,
that is the intended lesson, not a claim that CRC32 bags beat a
BiLSTM.
