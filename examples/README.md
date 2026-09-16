# Examples

Runnable scripts for the committed `dataset/` files. They do **not** load
GloVe, emoji2vec, or the Keras checkpoints.

```text
examples/
  sarcasm_lab/          shared library (tokenize, cues, NB, attention)
  dataset_overview.py   split sizes, lengths, mention / url rates
  cue_analysis.py       hashtag / emoji / phrase vs label
  lexical_baseline.py   bag-of-words NB + SGD logistic
  attention_walkthrough.py
  tokenize_demo.py      side-by-side with a few committed tweets
```

From the repo root:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/dataset_overview.py
python3 examples/cue_analysis.py
python3 examples/lexical_baseline.py
python3 examples/attention_walkthrough.py
python3 examples/tokenize_demo.py
```

`sarcasm_lab` is imported by adding the repo root and `examples/` to
`sys.path` inside each script, so you can also run a file by its absolute
path.

These scripts are for reading the personal course data. They are not a
Twitter client and they do not call any social API.
