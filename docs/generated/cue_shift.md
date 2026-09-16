# Cue shift

| split | cue | hits | coverage | cov_lo | cov_hi | hit_prec | odds_ratio |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| train | explicit | 3733 | 0.0938 | 0.0910 | 0.0967 | 0.9751 | 55.5895 |
| train | not_tag | 3202 | 0.0805 | 0.0779 | 0.0832 | 0.9738 | 50.9227 |
| train | sarcasm_tag | 532 | 0.0134 | 0.0123 | 0.0146 | 0.9831 | 65.2825 |
| train | emoji | 5514 | 0.1386 | 0.1353 | 0.1420 | 0.4164 | 0.7965 |
| train | contrast | 155 | 0.0039 | 0.0033 | 0.0046 | 0.8710 | 7.6610 |
| test | explicit | 615 | 0.3075 | 0.2877 | 0.3281 | 1.0000 | 3194.8521 |
| test | not_tag | 467 | 0.2335 | 0.2155 | 0.2525 | 1.0000 | 1753.4536 |
| test | sarcasm_tag | 150 | 0.0750 | 0.0643 | 0.0874 | 1.0000 | 354.0864 |
| test | emoji | 279 | 0.1395 | 0.1250 | 0.1554 | 0.6165 | 1.7305 |
| test | contrast | 38 | 0.0190 | 0.0139 | 0.0260 | 1.0000 | 80.0400 |
| subtest | explicit | 134 | 0.4820 | 0.4239 | 0.5406 | 1.0000 | 744.1169 |
| subtest | not_tag | 110 | 0.3957 | 0.3400 | 0.4542 | 1.0000 | 376.5840 |
| subtest | sarcasm_tag | 24 | 0.0863 | 0.0587 | 0.1252 | 1.0000 | 35.1414 |
| subtest | emoji | 278 | 1.0000 | 0.9864 | 1.0000 | 0.6187 | 1.6197 |
| subtest | contrast | 38 | 0.1367 | 0.1012 | 0.1821 | 1.0000 | 60.9703 |

Wilson intervals are 95%. Odds ratios use a 0.5 Haldane–Anscombe correction. `#not` coverage jumping from train to test is the main leakage number.
