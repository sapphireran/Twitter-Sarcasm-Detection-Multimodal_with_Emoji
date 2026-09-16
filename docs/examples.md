# Examples

All scripts run from the repository root:

```bash
python3 examples/<script>.py
```

They import `sarcasm_lib`. None of them load TensorFlow. Only
`inspect_emoji2vec.py` needs an extra package (Gensim).

| Script | What it proves |
| --- | --- |
| `inspect_dataset.py` | Split sizes, sarcasm rates, top hashtags / emoji, and that subtest ⊂ test |
| `tokenize_tweets.py` | CSV-aware vs legacy comma-splitting, plus the lightweight tokenizer |
| `heuristic_baseline.py` | Precision / recall of explicit cues on train, test, and subtest |
| `attention_walkthrough.py` | Raffel `α` weights on a three-token toy tweet |
| `predict_cli.py` | Score one tweet from the command line |
| `inspect_emoji2vec.py` | Neighbours plus `--report` coverage of corpus emoji vs the 200-d table |

Sample invocations are in `docs/reproduction.md` and the root `README.md`.
Each script accepts `-h`.
