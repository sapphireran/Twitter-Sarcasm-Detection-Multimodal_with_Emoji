# Archive lab examples

All scripts are meant to be run from the repository root:

```bash
PYTHONPATH=. python3 examples/01_split_census.py
```

or the whole suite:

```bash
PYTHONPATH=. python3 -m ccs2lab --skip-baseline
```

| Script | Reads | Writes to stdout |
| --- | --- | --- |
| `01_split_census.py` | `dataset/*.csv` | sizes, label balance, overlaps, length stats |
| `02_cue_shift.py` | same CSVs | cue coverage, Wilson intervals, odds ratios |
| `03_emoji2vec_probe.py` | `emoji2vec*.bin` | headers, nearest emoji, mean-pool demo |
| `04_attention_replay.py` | nothing on disk | Raffel identities (uniform, peak, mask) |
| `05_cue_rule.py` | CSVs | hashtag-rule scores vs recorded SVM |
| `06_hashed_baseline.py` | CSVs | hashed logistic + slice table |
| `07_reprint_scores.py` | `ccs2lab.recorded` | the 2023 notebook table |

`06_hashed_baseline.py` accepts `--train-limit N` for a faster fit
and `--no-cues` to drop the explicit cue features.

Generated markdown snapshots from a full run are copied under
`docs/generated/` so the docs stay in sync with the scripts.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

The tests cover tokenization, cue profiles, split integrity, the
word2vec header/reader, attention identities, metric edge cases, and
a tiny logistic fit on synthetic `#not` tweets.
