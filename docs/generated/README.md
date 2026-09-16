# Generated snapshots

These files were produced by the archive lab scripts on this clone.
Re-run the commands in [examples.md](../examples.md) to refresh them.

| File | Script |
| --- | --- |
| [split_census.md](split_census.md) | `examples/01_split_census.py` |
| [cue_shift.md](cue_shift.md) | `examples/02_cue_shift.py` |
| [emoji2vec_probe.md](emoji2vec_probe.md) | `examples/03_emoji2vec_probe.py` |
| [attention_replay.md](attention_replay.md) | `examples/04_attention_replay.py` |
| [cue_rule.md](cue_rule.md) | `examples/05_cue_rule.py` |
| [hashed_baseline.md](hashed_baseline.md) | `examples/06_hashed_baseline.py --train-limit 12000` |
| [recorded_scores.md](recorded_scores.md) | `examples/07_reprint_scores.py` |

The hashed logistic (stratified 12k train rows) landed at **0.773
accuracy on official test**, next to the recorded SVM 0.769. The
`explicit_cue` slice is 1.000; `no_explicit_cue` drops to 0.672
accuracy / 0.371 F1. That is the leakage split.
