# Lexical baseline

L2 logistic regression on hand-built cues from `examples/lib/lexical_features.py`.
Trained on train, evaluated on test and subtest. No embeddings.

## Metrics

| Split | accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.6647 | 0.6876 | 0.5105 | 0.5860 |
| test | 0.6395 | 0.6348 | 0.6570 | 0.6457 |
| subtest | 0.7806 | 0.8171 | 0.8314 | 0.8242 |

## Cue rates on train (binary features)

| feature | all | sarcastic | non-sarcastic |
| --- | ---: | ---: | ---: |
| `has_not_tag` | 0.080 | 0.168 | 0.004 |
| `has_elongation` | 0.139 | 0.160 | 0.121 |
| `has_love` | 0.113 | 0.129 | 0.100 |
| `has_positive_word_plus_neg_tag` | 0.021 | 0.045 | 0.001 |
| `has_great` | 0.041 | 0.037 | 0.044 |
| `has_yay` | 0.016 | 0.021 | 0.012 |
| `has_sarcastic_tag` | 0.007 | 0.015 | 0.000 |
| `has_love_when` | 0.007 | 0.013 | 0.002 |
| `has_face_unamused` | 0.008 | 0.013 | 0.004 |
| `has_sure` | 0.009 | 0.010 | 0.009 |
| `has_face_sweat` | 0.005 | 0.005 | 0.004 |
| `has_face_expressionless` | 0.003 | 0.005 | 0.002 |
| `has_so_fun` | 0.003 | 0.002 | 0.003 |
| `has_face_blank` | 0.002 | 0.002 | 0.002 |
| `has_as_if` | 0.001 | 0.001 | 0.001 |
| `has_just_great` | 0.000 | 0.001 | 0.000 |
| `has_oh_great` | 0.000 | 0.000 | 0.000 |
| `has_yeah_right` | 0.000 | 0.000 | 0.000 |
| `has_face_upside_down` | 0.000 | 0.000 | 0.000 |
| `has_face_rolling` | 0.000 | 0.000 | 0.000 |
| `has_sarcasm_tag` | 0.000 | 0.000 | 0.000 |

## Largest |weight| after standardization

| feature | weight |
| --- | ---: |
| `has_not_tag` | +1.181 |
| `n_mentions` | -0.475 |
| `n_tokens` | +0.434 |
| `has_sarcastic_tag` | +0.393 |
| `n_excl` | -0.215 |
| `n_hashtags` | -0.215 |
| `n_emoji` | -0.143 |
| `has_positive_word_plus_neg_tag` | +0.126 |
| `has_love_when` | +0.124 |
| `has_yay` | +0.111 |
| `n_chars` | -0.101 |
| `has_love` | +0.072 |
| `has_yeah_right` | +0.070 |
| `has_face_unamused` | +0.055 |
| `has_face_expressionless` | +0.049 |
| `has_elongation` | +0.045 |

Positive weight pushes the sarcastic class. `#not` / sarcasm hashtags
should dominate; emoji faces are weaker because they also appear on
sincere tweets.
