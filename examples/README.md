# Examples

Small scripts that explain this personal course snapshot without opening Jupyter or downloading GloVe. All of them are meant to be run from the **repository root**:

```bash
python examples/explore_dataset.py
```

Dependency: Python 3.10+ and `numpy` (already enough for every script here).

## What each script is for

| Script | Reads | Writes | Purpose |
| --- | --- | --- | --- |
| `explore_dataset.py` | `dataset/*` | stdout (or `--json`) | Split sizes, label balance, emoji / hashtag / mention rates, cue-tag leakage, length histograms. Companion to `docs/dataset.md`. |
| `tokenize_tweets.py` | curated rows + optional CLI strings | stdout | Shows comma collapse (`ReadOpen`) and a tweet-ish tokenizer vs naive whitespace. Companion to `docs/preprocessing.md`. |
| `inspect_emoji2vec.py` | `emoji2vec_twitter.bin`, optional `emoji2vec.bin` | stdout | Vocab, norms, nearest neighbours, 200-d vs 300-d warning. Companion to `docs/embeddings.md`. |
| `attention_demo.py` | nothing on disk | stdout | NumPy clone of `attention_layer.Attention` on toy clash sequences + invariant checks. Companion to `docs/architecture.md`. |
| `cue_baseline.py` | `dataset/*` | stdout | Majority / cue-tag / clash heuristics with live acc/P/R/F1. Companion to `docs/evaluation.md`. |
| `reprint_course_results.py` | nothing on disk | stdout | Prints the June 2023 notebook table. Historical; does not reload models. |
| `sample_annotations.md` | — | — | Hand-commented tweets (clash, sincere emoji, `#not` leakage). |
| `run_all.py` | the scripts above | stdout | Smoke-runs the lot. |

`common.py` is the shared loader: line-aligned sentence/label readers, a tiny tweet tokenizer, a word2vec-binary reader, and the classification scores used by the cue baseline.

## Useful flags

```bash
python examples/explore_dataset.py --split test --json
python examples/tokenize_tweets.py --synthetic-only
python examples/tokenize_tweets.py "wow I love exams 😒 #not"
python examples/inspect_emoji2vec.py --file both --neighbors 😒 😍 🌲
python examples/inspect_emoji2vec.py --compare
python examples/attention_demo.py --dim 16
python examples/cue_baseline.py --split test --errors 5 --error-rule clash
python examples/reprint_course_results.py --metric acc
python examples/run_all.py
```

## What these scripts will not do

- They will not load `model/best_model_*` (that needs TensorFlow, GloVe, and the train tokenizer).
- They will not fit sklearn pickles.
- They will not call any social-media API. Everything is local files from this repo.

For the full stack see `docs/training.md` and `requirements.txt`.
