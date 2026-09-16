# Examples

Run these from the repository root so `sarcasm_lib` imports resolve.

```bash
python3 examples/inspect_dataset.py
python3 examples/tokenize_tweets.py --split train --limit 5
python3 examples/heuristic_baseline.py
python3 examples/attention_walkthrough.py
python3 examples/predict_cli.py --file examples/sample_tweets.txt
```

`inspect_emoji2vec.py` is optional and needs Gensim:

```bash
python3 -m pip install gensim
python3 examples/inspect_emoji2vec.py --neighbors "😂"
```

Longer commentary for each script is in `docs/examples.md`.
