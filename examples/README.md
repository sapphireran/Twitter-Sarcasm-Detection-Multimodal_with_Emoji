# Personal examples

These scripts are a post-course walkthrough of the 2023 sarcasm project.
They read the checked-in CSVs and a frozen results table. They do **not**
load GloVe, TensorFlow, or the pickled sklearn models.

Run them from the repository root:

```bash
python3 -m examples.inspect_dataset
python3 -m examples.reported_results --all-metrics
python3 -m examples.tokenize_tweets --split subtest --limit 8
python3 -m examples.attention_walkthrough
python3 -m examples.multimodal_fusion
python3 -m examples.embedding_matrix_walkthrough
python3 -m examples.lexical_baseline
```

Or run everything through `python3 examples/run_all.py`.

## Scripts

| Command | What it shows |
| --- | --- |
| `inspect_dataset` | Split sizes, blocked labels, emoji / `#not` rates, and a proof that subtest is the emoji slice of test |
| `reported_results` | The June 2023 accuracy / F1 / precision / recall cards |
| `tokenize_tweets` | A TweetTokenizer-shaped split of real rows |
| `attention_walkthrough` | Raffel `α` on a 5-token toy sequence (`--mask-last` drops `#not`) |
| `multimodal_fusion` | Classical 200-d vs 400-d concat with synthetic tables |
| `embedding_matrix_walkthrough` | Shared Keras-style table, with and without emoji fallback |
| `lexical_baseline` | Logistic model on 16 surface features from the real CSVs |

## Library modules

| Module | Role |
| --- | --- |
| `dataset_io.py` | CSV loader, cue stats, subtest identity check |
| `tokenize.py` | Comma-join + lightweight tweet tokenizer |
| `lexical.py` | Surface feature schema |
| `logreg.py` | L2 logistic regression and binary scores |
| `attention_numpy.py` | NumPy port of `attention_layer.Attention` |
| `fusion.py` | Mean-pool concat and shared embedding matrix |
| `results_catalog.py` | Frozen 2023 numbers used by the docs |

## Lexical baseline flags

```bash
python3 -m examples.lexical_baseline --max-train 8000
python3 -m examples.lexical_baseline --drop-explicit-cues
```

`--drop-explicit-cues` zeros `#not` / `#sarcasm` / `#yeahright` so you can
see how much of a surface model is just those markers. The printed scores
are **not** the GloVe / Bi-LSTM numbers in `docs/experiments.md`.
