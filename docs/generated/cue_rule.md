# Cue rule baseline

## cue rule (predict sarcastic iff an explicit hashtag is present)
| split | slice | n | accuracy | precision | recall | f1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | all | 39780 | 0.6244 | 0.9751 | 0.1969 | 0.3276 |
| train | explicit_cue | 3733 | 0.9751 | 0.9751 | 1.0000 | 0.9874 |
| train | no_explicit_cue | 36047 | 0.5881 | 0.0000 | 0.0000 | 0.0000 |
| train | has_emoji | 5514 | 0.7078 | 0.9711 | 0.3075 | 0.4671 |
| train | no_emoji | 34266 | 0.6110 | 0.9760 | 0.1812 | 0.3057 |
| train | not_tag | 3202 | 0.9738 | 0.9738 | 1.0000 | 0.9867 |
| test | all | 2000 | 0.8075 | 1.0000 | 0.6150 | 0.7616 |
| test | explicit_cue | 615 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | no_explicit_cue | 1385 | 0.7220 | 0.0000 | 0.0000 | 0.0000 |
| test | has_emoji | 279 | 0.8638 | 1.0000 | 0.7791 | 0.8758 |
| test | no_emoji | 1721 | 0.7984 | 1.0000 | 0.5809 | 0.7349 |
| test | not_tag | 467 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| subtest | all | 278 | 0.8633 | 1.0000 | 0.7791 | 0.8758 |
| subtest | explicit_cue | 134 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| subtest | no_explicit_cue | 144 | 0.7361 | 0.0000 | 0.0000 | 0.0000 |
| subtest | has_emoji | 278 | 0.8633 | 1.0000 | 0.7791 | 0.8758 |
| subtest | not_tag | 110 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## recorded SVM (notebook outputs, not a re-run)
| split | modality | accuracy | f1 | precision | recall |
| --- | --- | ---: | ---: | ---: | ---: |
| test | W | 0.7690 | 0.7722 | 0.7617 | 0.7830 |
| test | WE | 0.7630 | 0.7663 | 0.7558 | 0.7770 |
| subtest | W | 0.8129 | 0.8523 | 0.8333 | 0.8721 |
| subtest | WE | 0.8237 | 0.8529 | 0.8820 | 0.8256 |

On test, the hashtag rule is a high-precision, moderate-recall classifier because `#not` / `#sarcasm` are dense. On the no_explicit_cue slice, recall is zero by construction. That gap is the part a sequence model still has to earn.
