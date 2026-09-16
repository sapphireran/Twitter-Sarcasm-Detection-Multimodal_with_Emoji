# Personal examples

Small scripts that run on the checked-in CSVs. They document the 2023 sarcasm
project without GloVe or TensorFlow.

| Command | What it does |
| --- | --- |
| `python3 -m examples.dataset_report` | Split sizes, cue rates, overlap |
| `python3 -m examples.emoji_signals` | Emoji log-odds of sarcasm |
| `python3 -m examples.tweet_tokenizer_demo` | Token + feature walkthrough |
| `python3 -m examples.attention_demo` | NumPy port of the LSTM attention |
| `python3 -m examples.metrics_table` | Recorded 2023 accuracy / F1 |
| `python3 -m examples.lexical_baseline_demo` | Tiny logistic model on lexical cues |

Helpers live in `examples/lib/`. Longer notes are in `docs/examples.md`.
