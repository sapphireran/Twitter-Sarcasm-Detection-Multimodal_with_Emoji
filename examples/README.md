# Examples

These scripts document the 2023 UCPH CCS2 sarcasm project using only
the CSVs already in ``dataset/``. They do **not** call Twitter/X, do
not need GloVe, and do not need TensorFlow.

Run them from the repo root (or from this folder; each script puts the
repo on ``sys.path``):

```bash
python examples/01_inspect_dataset.py
python examples/02_tokenize_tweets.py
python examples/03_attention_pooling.py
python examples/04_lexicon_baseline.py
python examples/05_cue_logistic.py
python examples/06_pipeline_walkthrough.py
python -m sarcasm_toolkit
```

| Script | What it shows |
| --- | --- |
| `01_inspect_dataset.py` | Split sizes, emoji/hashtag rates by label, top tags |
| `02_tokenize_tweets.py` | Hashtag / emoji / `<user>` tokenization |
| `03_attention_pooling.py` | Raffel attention vs mean pooling on toy vectors |
| `04_lexicon_baseline.py` | `#not` / `#sarcasm` floor vs majority class |
| `05_cue_logistic.py` | Stdlib logistic regression on cue features |
| `06_pipeline_walkthrough.py` | One-command path through all of the above |

`sample_tweets.csv` is a 10-row slice of patterns that show up in train
and subtest (contrast frames, cue hashtags, emoji-only literals).
`reported_results.json` is a typed copy of the executed notebook scores
so examples can print the 2023 BiLSTM numbers next to the new baselines.

The original experiment notebooks are unchanged:

* `baseline_models.ipynb` — SVM / DT / RF / GBT
* `evaluate_loaded_dl_models.ipynb` — load the saved BiLSTMs
* `get_metrics_of_models.ipynb` — accuracy / F1 / precision / recall
