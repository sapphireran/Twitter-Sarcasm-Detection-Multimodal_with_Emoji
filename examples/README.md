# Examples

Personal walkthroughs for the 2023 sarcasm-detection project. They
read only the checked-in CSVs plus NumPy. They do **not** load GloVe,
TensorFlow, or the sklearn pickles.

Run everything from the repository root:

```bash
python3 examples/inspect_dataset.py
python3 examples/preprocess_walkthrough.py
python3 examples/emoji_signal.py
python3 examples/heuristic_baseline.py
python3 examples/attention_demo.py
python3 examples/report_metrics.py
```

`python3 examples/run_all.py` executes the same set (via
`sys.executable`) and fails if any script exits non-zero.

## What each script is for

| Script | Question it answers |
| --- | --- |
| `inspect_dataset.py` | How big are the splits, and how skewed are the cues? |
| `preprocess_walkthrough.py` | What does `ReadOpen` + pad actually do to a tweet? |
| `emoji_signal.py` | Do emoji-like tokens move P(sarcastic)? |
| `heuristic_baseline.py` | How far does a `#not` / `#sarcasm` rule get you? |
| `attention_demo.py` | What do the Raffel attention weights look like? |
| `report_metrics.py` | What did the 2023 notebooks print? |

Shared code lives in `examples/lib/`:

- `dataset.py` — line-aligned sentence/label loader
- `tokenize.py` — comma unwrap + regex stand-in for `TweetTokenizer`

The regex tokenizer is documented as an approximation. The original
notebooks used `nltk.TweetTokenizer`. Counts will differ slightly
(especially around emoji clusters and punctuation).

## Suggested reading order

1. `inspect_dataset.py` then [../docs/dataset.md](../docs/dataset.md)
2. `preprocess_walkthrough.py` then [../docs/code-map.md](../docs/code-map.md)
3. `emoji_signal.py` + `heuristic_baseline.py` then
   [../docs/architecture.md](../docs/architecture.md)
4. `attention_demo.py` (the deep pooling step)
5. `report_metrics.py` then [../docs/results.md](../docs/results.md)

## Flags worth knowing

```bash
python3 examples/inspect_dataset.py --json
python3 examples/preprocess_walkthrough.py --split subtest --n 8
python3 examples/heuristic_baseline.py --show-errors --split train --n-errors 6
python3 examples/report_metrics.py --markdown
python3 examples/report_metrics.py --csv
```

## Tests

```bash
python3 -m unittest discover -s examples/tests -v
```
