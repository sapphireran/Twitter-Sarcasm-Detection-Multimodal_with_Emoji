# Hashed logistic baseline (stratified train-limit 12000)

fitted hashed logreg  n_train=12000  hash_dim=2048  cues=True  epochs=8  seed=7
cue weights (abs-sorted):
  explicit         +1.789
  not_tag          +1.355
  user             -0.971
  sarcasm_tag      +0.435
  exclaim          -0.378
  emoji            -0.294
  question         +0.101
  positive         +0.086
  contrast         +0.081
  elongation       +0.005

| split | slice | n | accuracy | precision | recall | f1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | all | 39780 | 0.6837 | 0.7424 | 0.4890 | 0.5896 |
| train | explicit_cue | 3733 | 0.9751 | 0.9751 | 1.0000 | 0.9874 |
| train | no_explicit_cue | 36047 | 0.6535 | 0.6396 | 0.3638 | 0.4637 |
| train | has_emoji | 5514 | 0.7089 | 0.9638 | 0.3127 | 0.4722 |
| train | no_emoji | 34266 | 0.6796 | 0.7280 | 0.5140 | 0.6026 |
| train | not_tag | 3202 | 0.9738 | 0.9738 | 1.0000 | 0.9867 |
| test | all | 2000 | 0.7730 | 0.7868 | 0.7490 | 0.7674 |
| test | explicit_cue | 615 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | no_explicit_cue | 1385 | 0.6722 | 0.3976 | 0.3481 | 0.3712 |
| test | has_emoji | 279 | 0.8674 | 1.0000 | 0.7849 | 0.8795 |
| test | no_emoji | 1721 | 0.7577 | 0.7515 | 0.7415 | 0.7465 |
| test | not_tag | 467 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| subtest | all | 278 | 0.8669 | 1.0000 | 0.7849 | 0.8795 |
| subtest | explicit_cue | 134 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| subtest | no_explicit_cue | 144 | 0.7431 | 1.0000 | 0.0263 | 0.0513 |
| subtest | has_emoji | 278 | 0.8669 | 1.0000 | 0.7849 | 0.8795 |
| subtest | not_tag | 110 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

This is not the 2023 SVM. If `not_tag` / `explicit` dominate the cue weights and the no_explicit_cue slice drops, the official test number is partly hashtag leakage.
