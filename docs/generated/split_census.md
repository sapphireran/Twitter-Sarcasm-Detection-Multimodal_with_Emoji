# Split census

| split | n | sarcastic | sarc_rate | sincere | mean_tok | median_tok | max_tok | min_tok |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| train | 39780 | 18488 | 0.465 | 21292 | 18.46 | 18.0000 | 92 | 1 |
| test | 2000 | 1000 | 0.500 | 1000 | 16.94 | 16.0000 | 41 | 1 |
| subtest | 278 | 172 | 0.619 | 106 | 17.94 | 17.0000 | 36 | 5 |

train ∩ test (unique strings): 48
test ∩ subtest: 278
train ∩ subtest: 0
subtest ⊆ test: True
expected sizes train=39780 test=2000 subtest=278
